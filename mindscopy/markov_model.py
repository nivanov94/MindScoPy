import numpy as np
from .cluster_base import UnsupervisedSegmentation
from .utils.transition_matrix import count_state_transitions
from .utils.visualization import plot_pmfs, show_transition_matrices
import matplotlib.pyplot as plt


class MarkovStateSpace(UnsupervisedSegmentation):
    """
    Model mental imagery trials using an unsupervised clustering-based
    segmentation of the EEG signal space and a Markov chain model [1]_.
    
    References
    ----------
    .. [1] Ivanov, N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    def __init__(
            self, clustering_model=None, n_clusters=None, 
            k_selection_thresh=0.3, krange=range(2, 12), prefit=False
    ):
        """
        Initialize the Markov Chain Model object.
        """
        super().__init__(clustering_model, n_clusters, k_selection_thresh, krange, prefit)
        self.n_states = None


    def fit(self, X, y=None, verbose=False):
        """
        Perform the clustering-based segmentation of the EEG signal space and
        to define the Markov chain model's states.

        Parameters
        ----------
        X : array_like (n_trials, n_epochs, Nf)
            The input signal to be segmented, where n_trials is the number of trials,
            n_epochs is the number of epochs and Nf is the number of features.

        y : array_like (n_trials,)
            The ground truth labels for the trials. If provided and using
            prediction strength for k selection, the labels will be used to
            split the data into training and testing sets.

        verbose : bool, optional
            Whether to print the number of clusters selected. Default is False.
        """

        self.fit_cluster_model(X, y=y, verbose=verbose)
        if hasattr(self.clustering_model, 'cluster_centers_'):
            self.n_states = self.clustering_model.cluster_centers_.shape[0]
        else:
            raise ValueError("Clustering model does not have cluster_centers_ attribute.")
        return self
    
    def transform(self, X, y=None):
        """
        Transform the EEG signal space into the Markov chain model's state space.

        Parameters
        ----------
        X : array_like (n_trials, n_epochs, n_feats)
            The input signal to be segmented, where n_trials is the number of trials,
            n_epochs is the number of epochs and Nf is the number of features.
        y : array_like (n_trials,), optional
            Unused parameter. Default is None.

        Returns
        -------
        S : array_like (n_trials, n_epochs)
            The state sequence of the Markov chain model.
        """
        if len(X.shape) != 3:
            raise ValueError("Input data must be 3-dimensional (n_trials, n_epochs, n_features)")

        n_trials, n_epochs, n_feats = X.shape
        X = np.reshape(X, (n_trials*n_epochs, n_feats))  # stack first 2 dims
        S = self.clustering_model.predict(X)
        S = np.reshape(S, (n_trials, n_epochs))
        return S


class MarkovChainModel:
    """
    Markov chain representation of clustered EEG state transitions [1]_.
    
    References
    ----------
    .. [1] Ivanov, N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    def __init__(self, cluster_mdl):
        """
        Initialize the Markov Chain Model object.
        """
        self.cluster_mdl = cluster_mdl
        self.n_states = cluster_mdl.n_states

        if self.n_states is None:
            raise ValueError("The clustering model must be fitted and have a integer n_states attribute.")

        self.transition_matrix = None
        self.steady_state = None
        self.entropy_rate = None

    def fit(self, S, damping=0.05, verbose=False):
        """
        Fit the Markov chain model to the state sequence S.

        Parameters
        ----------
        S : array_like (n_trials, n_epochs)
            The state sequence of the Markov chain model. n_trials is the number of trials
            and n_epochs is the number of epochs per trial.
        damping : float, optional
            The damping factor for the transition matrix. Default is 0.05.
        verbose : bool, optional
            Whether to print the transition matrix. Default is False.
        """
        if S.ndim != 2:
            raise ValueError("Input state sequence S must be 2-dimensional (n_trials, n_epochs).")

        n_trials, n_epochs = S.shape
        transition_matrix = count_state_transitions(S, self.n_states)

        # apply row-wise damping to the transition counts
        total_trans_count = n_trials * n_epochs
        for i in range(self.n_states):
            # compute the raw observation frequencies
            state_trans_count = np.sum(transition_matrix[i])
            if state_trans_count > 0:
                transition_matrix[i] /= state_trans_count

            # dampen the row based on the total number of transitions
            # observed from the state compared to the total number of
            # transitions observed
            state_prop = state_trans_count / total_trans_count
            transition_matrix[i] = (
                min(state_prop*self.n_states, 1) * transition_matrix[i] +
                max(1 - state_prop*self.n_states, 0) * np.ones((self.n_states,)) / self.n_states
            )

        # apply damping to the entire transition matrix
        # to ensure that the matrix is irreducible and aperiodic
        self.transition_matrix = (
            (1 - damping) * transition_matrix
            + damping * np.ones((self.n_states, self.n_states)) / self.n_states
        )

        if verbose:
            print(f'Transition matrix:\n{np.array2string(self.transition_matrix, formatter={"float_kind":lambda x: "%.3f" % x})}')

        # compute the steady state distribution and the entropy rate
        self.compute_steady_state()
        self.compute_entropy_rate()

        return self

    
    def compute_steady_state(self):
        """
        Compute the steady state distribution of the Markov chain model.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix is not defined. Fit the model first.")

        # compute the eigenvalues and eigenvectors of the transition matrix
        w, v = np.linalg.eig(self.transition_matrix.T)

        # extract the eigenvector corresponding to the largest eigenvalue (equal to 1)
        v = np.real(v[:, np.isclose(w, 1)])[:,0]
        if v[0] < 0:
            v = -v
        self.steady_state = v / np.sum(v)  # normalize the eigenvector to sum to 1

    def compute_entropy_rate(self):
        """
        Compute the entropy rate of the Markov chain model.
        """
        if self.transition_matrix is None:
            raise ValueError("Transition matrix is not defined. Fit the model first.")

        # compute the state transition entropy
        A = np.clip(self.transition_matrix, 1e-12, 1)  # avoid log(0)
        state_entropy = -np.sum(A * np.log2(A), axis=1)
        self.entropy_rate = np.dot(self.steady_state, state_entropy)
    

def hellinger_distance(P, Q):
    """
    Compute the Hellinger distance between two probability distributions P and Q.

    Parameters
    ----------
    P : array_like (N,)
        The first probability distribution.
    Q : array_like (N,)
        The second probability distribution.

    Returns
    -------
    hellinger : float
        The Hellinger distance between the two probability distributions.
    """
    P, Q = np.asarray(P, dtype=np.float64), np.asarray(Q, dtype=np.float64)  # ensure inputs are numpy arrays
    if P.shape != Q.shape:
        raise ValueError("Input probability distributions must have the same shape.")

    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2))


def task_distinct(markov_models, visualize=False, mode='pairwise'):
    """
    Compute the taskDistinct metric from [1]_ 
    for a set of Markov chain models.
    
    Parameters
    ----------
    markov_models : list
        A list of Markov chain models fitted to different tasks.
    
    visualize : bool, optional
        Whether to visualize the steady state distributions of the models.
        Default is False.

    mode : str, optional
        The mode of taskDistinct computation. Options are:
        - 'pairwise': average pairwise Hellinger distance between the steady state distributions.
        - 'mean': average Hellinger distance between each model and the mean steady state distribution.
        - 'distance_to_closest': average Hellinger distance between each model and the closest other model.
        Default is 'pairwise'.

    Returns
    -------
    task_distinct : float
        The taskDistinct metric.

    References
    ----------
    .. [1] Ivanov, N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """
    if len(markov_models) < 2:
        raise ValueError("At least two markov_models are required to compute task_distinct")
    
    
    n_models = len(markov_models)
    n_states = markov_models[0].n_states
    for m in markov_models:
        if m.n_states != n_states:
            raise ValueError("All models must have the same number of states")
    
    pmfs = np.stack([m.steady_state for m in markov_models])
    
    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        plot_pmfs(pmfs, ax, legend_str='Task', y_label='P(State | Task)')
    
    return _compute_task_distinct_from_pmfs(pmfs, mode)
    
    
def _compute_task_distinct_from_pmfs(pmfs, mode='pairwise'):
    """
    Compute the taskDistinct metric from the steady state distributions.
    """
    n_models, n_states = pmfs.shape
    task_distinct = 0.0
    
    if mode == 'pairwise':
        count = 0
        for i in range(n_models - 1):
            for j in range(i + 1, n_models):
                task_distinct += hellinger_distance(pmfs[i], pmfs[j])
                count += 1
        task_distinct /= max(1, count)
    
    elif mode == 'mean':
        avg = pmfs.mean(axis=0)
        for i in range(n_models):
            task_distinct += hellinger_distance(pmfs[i], avg)
        task_distinct /= n_models

    elif mode == 'distance_to_closest':
        for i in range(n_models):
            min_d = np.inf
            for j in range(n_models):
                if i == j:
                    continue
                d = hellinger_distance(pmfs[i], pmfs[j])
                if d < min_d:
                    min_d = d
            task_distinct += min_d
        task_distinct /= n_models

    else:
        raise ValueError(f"Unknown mode '{mode}' for task_distinct")
    
    return task_distinct


def relative_task_inconsistency(task_markov_models, rest_markov_model, visualize=False):
    """
    Compute the relativeTaskInconsistency metric from [1]_ 
    for a set of task Markov chain models and a resting state Markov chain
    model.
    
    Parameters
    ----------
    task_markov_models : list
        A list of Markov chain models fitted to different tasks.
    rest_markov_model : Markov_Model
        A Markov chain model fitted to the rest task.

    Returns
    -------
    relative_task_inconsistency : float
        The relative task inconsistency metric.

    References
    ----------
    .. [1] Ivanov, N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """
    if len(task_markov_models) == 0:
        raise ValueError("task_markov_models must not be empty")

    for m in task_markov_models:
        if m.entropy_rate is None:
            raise ValueError("All task models must have entropy_rate computed (call fit())")
    if rest_markov_model.entropy_rate is None:
        raise ValueError("rest_markov_model must have entropy_rate computed (call fit())")

    n_models = len(task_markov_models)

    if visualize:
        # plot the transition matrices of the models 
        # with the probability of each state transition as
        # a color-coded matrix
        fig, ax = plt.subplots(1, n_models+1, figsize=(5*n_models, 5))
        for i in range(n_models):

            title = f'Task {i+1}: H = {task_markov_models[i].entropy_rate:.3f}'
            show_transition_matrices(
                task_markov_models[i].transition_matrix, ax[i], title=title
            )
        
        show_transition_matrices(
            rest_markov_model.transition_matrix,
            ax[-1],
            title=f'Rest: H = {rest_markov_model.entropy_rate:.3f}'
        )
        plt.show()


    rel_task_incon = 0

    for i in range(n_models):
        rel_task_incon += task_markov_models[i].entropy_rate

    rel_task_incon /= n_models
    rel_task_incon += 1
    rel_task_incon /= (1 + rest_markov_model.entropy_rate)

    return rel_task_incon