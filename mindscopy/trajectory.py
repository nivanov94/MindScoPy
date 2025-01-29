import numpy as np
import pyriemann
from sklearn.cluster import KMeans
from .utils.cluster_identification import cluster_pred_strength


class Trajectory_Subspace:

    def __init__(self, clustering_model=None, n_clusters=None, k_selection_thresh=0.3, krange=range(2, 12)):
        self.clustering_model = clustering_model
        self.n_clusters = n_clusters
        self.k_selection_thresh = k_selection_thresh
        self.krange = krange
        self.subspace_bases = None
        self.ref = None # the reference point that acts as the origin for the subspace


    def fit(self, X):

        # if a clustering model is not provided, use KMeans
        if self.clustering_model is None:
            # determine the number of clusters in the model
            if self.n_clusters is not None:
                K = self.n_clusters
            else:
                K = self._select_k(X)

            self.clustering_model = KMeans(n_clusters=K)

        # fit the clustering model
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        self.clustering_model.fit(X)

        ## define the subspace spanned by the cluster centers
        # compute the mean of the cluster centers
        self.ref = np.mean(self.clustering_model.cluster_centers_, axis=0)

        # compute the subspace bases
        w = self.clustering_model.cluster_centers_ - self.ref

        # compute the SVD of the cluster centers
        U, S, V = np.linalg.svd(w, full_matrices=False)

        # extract the subspace bases
        for i in range(len(S)):
            if S[i] < 1e-10:
                break
        
        self.subspace_bases = U[:, :i]
        self._subspace_dim = i

    
    def transform(self, X, y=None):
        
        # project the data onto the subspace
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        X = X - self.ref
        X = np.dot(X, self.subspace_bases)

        return X
    
    def _select_k(self, X):
        
        # compute the prediction strength of clustering
        crit = cluster_pred_strength(
            X, thresh=self.k_selection_thresh, krange=self.krange
        )
        
        # select the number of clusters
        return np.argmax(crit) + min(self.krange)
    