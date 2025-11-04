import numpy as np
from sklearn.cluster import KMeans
from .utils.cluster_identification import cluster_pred_strength
from .utils.visualization import visualize_CSP
import pyriemann
import scipy
import matplotlib.pyplot as plt
import warnings


class UnsupervisedSegmentation:
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
            k_selection_thresh=0.3, krange=range(2, 12), prefit=False
    ):
        """
        Initialize the Markov Chain Model object.
        """
        self.clustering_model = clustering_model
        self.n_clusters = n_clusters
        self.k_selection_thresh = k_selection_thresh
        self.krange = tuple(krange) # conversion to allow non-contiguous ranges
        self.prefit = prefit


    def fit_cluster_model(self, X, y=None, verbose=False):
        """
        Perform the clustering-based segmentation of the EEG signal space and
        to define the Markov chain model's states.

        Parameters
        ----------
        X : array_like (n_trials, n_epochs, Nf)
            The input signal to be segmented, where n_trials is the number of trials,
            n_epochs is the number of sub-epochs and Nf is the number of features.

        y : array_like (n_trials,)
            The ground truth labels for the trials. If provided and using
            prediction strength for k selection, the labels will be used to
            split the data into training and testing sets.

        verbose : bool, optional
            Whether to print the number of clusters selected. Default is False.

        """
        if self.prefit:
            print("The clustering model is already fitted. Skipping fitting.")
            return

        if X.ndim != 3:
            raise ValueError("Input data X must be 3-dimensional (n_trials, n_epochs, Nf).")

        n_channels, n_epochs, Nf = X.shape
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        if y is not None:
            y = np.repeat(y, n_epochs)

        # if a clustering model is not provided, use KMeans
        if self.clustering_model is None:
            # determine the number of clusters in the model
            if self.n_clusters is None:
                if verbose:
                    print('Selecting the number of clusters using prediction strength.')
                self.n_clusters = self._select_k(X, y, verbose=verbose)
                if verbose:
                    print(f'Number of clusters selected: {self.n_clusters}')

            K = self.n_clusters
            self.clustering_model = KMeans(n_clusters=K)

        # fit the clustering model
        self.clustering_model.fit(X)


    def _select_k(self, X, y=None, verbose=False):
        
        # compute the prediction strength of clustering
        crit = cluster_pred_strength(
            X, y=y, krange=self.krange
        )
        if verbose:
            print(f"Prediction strength criterion for k selection:\n{crit}")

        # select the number of clusters, largest k with crit >= threshold
        k_sel = min(self.krange)
        if not np.any(crit >= self.k_selection_thresh):
            warnings.warn(
                "No k in the specified range met the selection threshold. "
                "Selecting the smallest k in the range."
            )
        else:
            for k, crit_k in zip(self.krange, crit):
                if crit_k >= self.k_selection_thresh and k > k_sel:
                    k_sel = k
       
        return k_sel
    

    def predict(self, X, y=None):
        """
        Assign each epoch in the input data to a state based on the 
        clustering model.

        Parameters
        ----------
        X : array_like (n_trials, n_epochs, n_feats)
            The input signal to be segmented, where n_trials is the number of trials,
            n_epochs is the number of epochs and n_feats is the number of features.

        y : array_like (n_trials,), optional
            Unused parameter. Default is None.

        Returns
        -------
        S : array_like (n_trials, n_epochs)
            The state sequence of the Markov chain model.
        """
        n_trials, n_epochs, _ = X.shape
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        clust_assgn = self.clustering_model.predict(X)
        S = np.reshape(clust_assgn, (n_trials, n_epochs))
        return S


    def plot_activation_patterns(self, X, Xcovs, chs):
        """
        Generate CSP activation pattern topographical plots for each state.
        Rather than using the arithmetic of the state covariance matrices as in
        [1]_, we use the Riemannian mean of each state as described in 
        [2]_.

        Parameters
        ----------
        X : array_like (n_trials, Nf)
            Corresponding feature vectors for the data. n_trials is the number of trials
            or epochs, Nf is the number of features. Each trial will be assigned to
            a state based on the clustering model.

        Xcovs : array_like (n_trials, n_channels, n_channels)
            The data with which to compute the activation patterns. n_trials is the number
            of trials or epochs, n_channels is the number of channels. Each trial will be 
            assigned to a state based on the clustering model.

        chs : list of str
            The MNE channel names for the EEG data.

        References
        ----------
        .. [1] Haufe, S. et al. (2014). On the interpretation of weight vectors of
               linear models in multivariate neuroimaging. NeuroImage, 87, 96-110.

        .. [2] Ivanov, N., Lio. A, and Chau, T. (2023) Towards user-centric BCI design:
                Markov chain-based user assessment for mental imagery EEG-BCIs.
                Journal of Neural Engineering, 20(6). 
        """

        clust_assgn = self.clustering_model.predict(X)
        n_patterns = self.clustering_model.cluster_centers_.shape[0]

        patterns = np.zeros(
            (n_patterns, Xcovs.shape[1], Xcovs.shape[1])
        )

        # across class mean
        all_class_mean = pyriemann.utils.mean.mean_riemann(Xcovs)

        for i_l, c in enumerate(np.sort(np.unique(clust_assgn))):
            # within class mean
            class_mean = pyriemann.utils.mean.mean_riemann(Xcovs[c==clust_assgn])

            # compute CSP filters
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
        fig, axes = plt.subplots(
            nrows=1, ncols=patterns.shape[0], figsize=(fwidth, 4)
        )

        if patterns.shape[0] == 1:
            axes = [axes]

        for i_p, pattern in enumerate(patterns):
            # visualize the patterns
            visualize_CSP(
                pattern/v_max, chs, -1, 1, 
                fig, axes[i_p], fwidth, cbar=(i_p==(patterns.shape[0]-1))
            )
            axes[i_p].set_title(f"State {i_p+1}", fontsize=10)
        plt.show()