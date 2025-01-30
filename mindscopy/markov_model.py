import numpy as np
from .cluster_base import Unsupervised_Segmentation
from .utils.transition_matrix import count_state_transitions


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
        self.n_states = cluster_mdl.cluster_centers_.shape[0]

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


        return self

    
