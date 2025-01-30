import numpy as np
from sklearn.cluster import KMeans
from .utils.cluster_identification import cluster_pred_strength
from .utils.visualization import visualize_CSP
import pyriemann
import scipy
import matplotlib.pyplot as plt


class Unsupervised_Segmentation:
    """
    Base class used for modeling mental imagery trials using 
    an unsupervised clustering-based segmentation of the EEG
    signal space [1]_.
    
    References
    ----------
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """

    def __init__(
            self, clustering_model=None, n_clusters=None, 
            k_selection_thresh=0.3, krange=(2, 12)
    ):
        """
        Initialize the Markov Chain Model object.
        """
        self.clustering_model = clustering_model
        self.n_clusters = n_clusters
        self.k_selection_thresh = k_selection_thresh
        self.krange = krange

    def fit_cluster_model(self, X, verbose=False):
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

        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))

        # if a clustering model is not provided, use KMeans
        if self.clustering_model is None:
            # determine the number of clusters in the model
            if self.n_clusters is not None:
                K = self.n_clusters
            else:
                if verbose:
                    print('Selecting the number of clusters using prediction strength.')
                K = self._select_k(X)
                if verbose:
                    print(f'Number of clusters selected: {K}')

            self.clustering_model = KMeans(n_clusters=K)

        # fit the clustering model
        self.clustering_model.fit(X)

    def _select_k(self, X):
        
        # compute the prediction strength of clustering
        crit = cluster_pred_strength(
            X, thresh=self.k_selection_thresh, krange=self.krange
        )
        
        # select the number of clusters
        K = min(self.krange)
        for i in range(len(crit)):
            if crit[i] >= self.k_selection_thresh:
                K = i + min(self.krange)
        return K
    
    def plot_state_activation_patterns(self, X, Xcovs, chs):
        """
        Generate CSP activation pattern topographical plots for each state.
        Rather than using the arithmetic of the state covariance matrices as in
        [1]_, we use the Riemannian mean of each state as described in 
        [2]_.

        Parameters
        ----------
        X : array_like (Nt, Nf)
            Corresponding feature vectors for the data. Nt is the number of trials
            or epochs, Nf is the number of features. Each trial will be assigned to
            a state based on the clustering model.

        Xcovs : array_like (Nt, Nc, Nc)
            The data with which to compute the activation patterns. Nt is the number
            of trials or epochs, Nc is the number of channels. Each trial will be 
            assigned to a state based on the clustering model.

        chs : list of str
            The channel names for the EEG data.

        References
        ----------
        .. [1] Haufe, S. et al. (2014). On the interpretation of weight vectors of
               linear models in multivariate neuroimaging. NeuroImage, 87, 96-110.

        .. [2] Ivanov, N., Lio. A, and Chau, T. (2023) Towards user-centric BCI design:
                Markov chain-based user assessment for mental imagery EEG-BCIs.
                Journal of Neural Engineering, 20(6). 
        """

        clust_assgn = self.clustering_model.predict(X)

        patterns = np.zeros(
            (self.clustering_model.cluster_centers_.shape[0], Xcovs.shape[1], Xcovs.shape[1])
        )

        all_class_mean = pyriemann.utils.mean.mean_riemann(Xcovs)
        for i_l, c in enumerate(np.unique(clust_assgn)):
            class_mean = pyriemann.utils.mean.mean_riemann(Xcovs[c==clust_assgn])
            l, V = scipy.linalg.eigh(class_mean, all_class_mean)

            # sort the eigenvalues and eigenvectors in order
            ix = np.flip(np.argsort(l)) 
    
            l = l[ix]
            V = V[:,ix]
    
            # convert the filters to the activation patterns
            A = scipy.linalg.pinv(V)
            patterns[i_l] = np.abs(A.T) # use the absolute values because these are variance-based activations
    
        v_max = np.max(patterns)
        fwidth = 2.5*patterns.shape[0]
        fig, axes = plt.subplots(nrows=1,ncols=patterns.shape[0], figsize=(fwidth, 4))

        for i_p, pattern in enumerate(patterns):
            # visualize the patterns
            visualize_CSP(pattern/v_max, chs, -1, 1, fig, axes[i_p], fwidth, cbar=(i_p==(patterns.shape[0]-1)))
        plt.show()