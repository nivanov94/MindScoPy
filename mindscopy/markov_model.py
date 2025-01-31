import numpy as np
from .cluster_base import Unsupervised_Segmentation
from .utils.transition_matrix import count_state_transitions
import matplotlib.pyplot as plt


class Markov_State_Space(Unsupervised_Segmentation):
    """
    Model mental imagery trials using an unsupervised clustering-based
    segmentation of the EEG signal space and a Markov chain model [1]_.
    
    References
    ----------
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    def __init__(
            self, clustering_model=None, n_clusters=None, 
            k_selection_threshold=0.3, krange=(2, 12)
    ):
        """
        Initialize the Markov Chain Model object.
        """
        super().__init__(clustering_model, n_clusters, k_selection_threshold, krange)


    def fit(self, X, verbose=False):
        """
        Perform the clustering-based segmentation of the EEG signal space and
        to define the Markov chain model's states.

        Parameters
        ----------
        X : array_like (Nt, Ne, Nf)
            The input signal to be segmented, where Nt is the number of trials,
            Ne is the number of sub-epochs and Nf is the number of features.

        verbose : bool, optional
            Whether to print the number of clusters selected. Default is False.
        """

        self.fit_cluster_model(X, verbose)
        return self
    
    def transform(self, X, y=None):
        """
        Transform the EEG signal space into the Markov chain model's state space.

        Parameters
        ----------
        X : array_like (Nt, Ne, Nf)
            The input signal to be segmented, where Nt is the number of trials,
            Ne is the number of sub-epochs and Nf is the number of features.
        y : array_like (Nt,), optional
            Unused parameter. Default is None.

        Returns
        -------
        S : array_like (Nt,)
            The state sequence of the Markov chain model.
        """
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        return self.clustering_model.predict(X)


class Markov_Model:
    """
    Model mental imagery trials using a Markov chain model [1]_.
    
    References
    ----------
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    def __init__(self, cluster_mdl):
        """
        Initialize the Markov Chain Model object.
        """
        self.cluster_mdl = cluster_mdl
        self.n_states = cluster_mdl.clustering_model.cluster_centers_.shape[0]

        self.transition_matrix = None
        self.steady_state = None
        self.entropy_rate = None

    def fit(self, S, damping=0.05, verbose=False):
        """
        Fit the Markov chain model to the state sequence S.

        Parameters
        ----------
        S : array_like (Nt, Ne)
            The state sequence of the Markov chain model. Nt is the number of trials
            and Ne is the number of sub-epochs per trial.
        damping : float, optional
            The damping factor for the transition matrix. Default is 0.05.
        verbose : bool, optional
            Whether to print the transition matrix. Default is False.
        """
        Nt, Ne = S.shape
        transition_matrix = count_state_transitions(S, self.n_states)

        # apply row-wise damping to the transition counts
        total_trans_count = Nt * Ne
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
            (1 - damping) * transition_matrix + 
            damping * np.ones((self.n_states, self.n_states)) / self.n_states
        )


        if verbose:
            print('Transition matrix:')
            print(self.transition_matrix)

        # compute the steady state distribution and the entropy rate
        self.compute_steady_state()
        self.compute_entropy_rate()

        return self

    
    def compute_steady_state(self):
        """
        Compute the steady state distribution of the Markov chain model.
        """
        # compute the eigenvalues and eigenvectors of the transition matrix
        w, v = np.linalg.eig(self.transition_matrix.T)

        # extract the eigenvector corresponding to the largest eigenvalue (equal to 1)
        w = np.abs(w)
        v = np.abs(v)
        self.steady_state = v[:, np.argmax(w)] / np.sum(v[:, np.argmax(w)]) # normalize the eigenvector to sum to 1

    def compute_entropy_rate(self):
        """
        Compute the entropy rate of the Markov chain model.
        """
        
        # compute the state transition entropy
        # we don't need to check for zeros since the transition matrix is already damped
        state_entropy = -np.sum(self.transition_matrix * np.log2(self.transition_matrix), axis=1)
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
    return np.sqrt(0.5 * np.sum((np.sqrt(P) - np.sqrt(Q))**2))


def task_distinct(markov_models, visualize=False):
    """
    Compute the taskDistinct metric from [1]_ 
    for a set of Markov chain models.
    
    Parameters
    ----------
    markov_models : list
        A list of Markov chain models fitted to different tasks.

    Returns
    -------
    task_distinct : float
        The taskDistinct metric.

    References
    ----------
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """
    from scipy.special import comb

    n_models = len(markov_models)
    n_states = markov_models[0].n_states

    if visualize:
        # plot the steady state distributions of the models
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        for i in range(n_models):
            ax.fill_between(np.arange(n_states)+1, markov_models[i].steady_state, step='pre', alpha=0.3, label=f'Task model {i+1}')
            ax.step(np.arange(n_states), markov_models[i].steady_state, where='pre')
        ax.set_xlabel('State')
        ax.set_ylabel('P(State | Task)')
        ax.set_ylim(0, 1)
        ax.set_xlim(0, n_states)
        ax.set_xticks(np.arange(n_states)+.5)
        ax.set_xticklabels(np.arange(n_states)+1)
        ax.legend()
        plt.show()


    task_distinct = 0

    for i in range(n_models-1):
        for j in range(i+1, n_models):
            task_distinct += hellinger_distance(markov_models[i].steady_state, markov_models[j].steady_state)

    # normalize the taskDistinct metric by the number of model pairs - binomial coefficient
    n_pairs = comb(n_models, 2)
    return task_distinct / n_pairs


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
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    n_models = len(task_markov_models)

    if visualize:
        # plot the transition matrices of the models 
        # with the probability of each state transition as
        # a color-coded matrix
        fig, ax = plt.subplots(1, n_models+1, figsize=(5*n_models, 5))
        viz_models = task_markov_models + [rest_markov_model]
        for i in range(len(viz_models)):
            ax[i].imshow(viz_models[i].transition_matrix, cmap='Greens', vmin=0, vmax=1)

            # add the probability values to the matrix
            for j in range(viz_models[i].n_states):
                for k in range(viz_models[i].n_states):
                    ax[i].text(k, j, f'{viz_models[i].transition_matrix[j, k]:.2f}', ha='center', va='center', color='black')

            if i == n_models:
                ax[i].set_title(f'Rest model: H = {rest_markov_model.entropy_rate:.3f}')
            else:
                ax[i].set_title(f'Task model {i+1}: H = {task_markov_models[i].entropy_rate:.3f}')
            ax[i].set_xlabel('Destination state')
            ax[i].set_ylabel('Origin state')
        
        plt.show()


    rel_task_incon = 0

    for i in range(n_models):
        rel_task_incon += task_markov_models[i].entropy_rate

    rel_task_incon /= n_models
    rel_task_incon += 1
    rel_task_incon /= (1 + rest_markov_model.entropy_rate)

    return rel_task_incon