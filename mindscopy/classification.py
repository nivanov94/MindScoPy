import numpy as np
from itertools import combinations
from scipy.linalg import eigh, pinv
from scipy.special import binom
from scipy import stats
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from matplotlib import pyplot as plt
import mne
import pyriemann


class CSP:
    """
    Common Spatial Pattern (CSP) implementation [1]_.

    Supports binary and multi-class (OVR or pairwise) extensions using either 
    Euclidean or Riemannian mean covariance matrices.

    Parameters
    ----------
    m : int, default=2
        Number of spatial filters per class.
    multi_class_mode : {'OVR', 'PW'}, default='OVR'
        Multi-class strategy: one-vs-rest (OVR) or pairwise (PW).
    mean_method : {'euclid', 'riem'}, default='euclid'
        Method for computing class-mean covariance matrices.
    log_var_feats : bool, default=False
        Whether to apply log-variance transformation to CSP features.

    References
    ----------
    .. [1] Koles, Z. J., Lind, J. C., & Flor-Henry, P. (1994). 
           Spatial patterns in the background EEG underlying mental disease 
           in man. Electroencephalography and clinical neurophysiology, 91(5), 
           319-328.
    """

    def __init__(
        self, m=2, multi_class_mode='OVR',
        mean_method='euclid', log_var_feats=False
    ):
        self.m = m
        self.multi_class_mode = multi_class_mode
        self.mean_method = mean_method
        self.log_var_feats = log_var_feats
        self.filters_ = None
        self.patterns_ = None


    def fit(self, X, y):
        """
        Fit CSP filters to data.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data.
        y : ndarray, shape (n_trials,)
            Labels.

        Returns
        -------
        self : CSP instance
            Fitted CSP instance.
        """
        W, A = self._compute_filters(X, y)
        self.filters_ = W
        self.patterns_ = A
        return self


    def transform(self, X):
        """
        Apply learned CSP filters to new data.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data.

        Returns
        -------
        X_csp : ndarray, shape (n_trials, n_filters) or (n_trials, n_filters, n_samples)
            Transformed CSP features.
        """
        Xcsp = self._apply_filters(self.filters_, X)
        return self.log_var(Xcsp) if self.log_var_feats else Xcsp


    def fit_transform(self, X, y):
        """
        Fit CSP and transform in one step.
        
        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data.
        y : ndarray, shape (n_trials,)
            Labels.

        Returns
        -------
        X_csp : ndarray, shape (n_trials, n_filters) or (n_trials, n_filters, n_samples)
            Transformed CSP features.
        """
        self.fit(X, y)
        return self.transform(X)


    def _compute_filters(self, X, y):
        """Compute CSP filters for binary or multi-class data."""
        n_classes = len(np.unique(y))
        if n_classes == 2:
            return self._compute_binary_filters(X, y)

        labels = np.unique(y)
        n_labels = len(labels)
        n_channels = X.shape[1]

        if self.multi_class_mode == 'OVR':
            W = np.zeros((n_labels, n_channels, 2 * self.m))
            A = np.zeros((n_labels, n_channels, n_channels))
            for i, lbl in enumerate(labels):
                yl = np.where(y == lbl, 0, 1)
                W[i], A[i] = self._compute_binary_filters(X, yl)

        elif self.multi_class_mode == 'PW':
            n_pairs = int(binom(n_labels, 2))
            W = np.zeros((n_pairs, n_channels, 2 * self.m))
            A = np.zeros((n_pairs, n_channels, n_channels))
            for i, (l1, l2) in enumerate(combinations(labels, 2)):
                mask = np.isin(y, [l1, l2])
                X_pair, y_pair = X[mask], y[mask]
                y_pair = np.where(y_pair == l1, 0, 1)
                W[i], A[i] = self._compute_binary_filters(X_pair, y_pair)

        else:
            raise ValueError("multi_class_mode must be 'OVR' or 'PW'.")
        
        return W, A


    def _compute_binary_filters(self, X, y):
        """Compute binary CSP spatial filters."""
        labels = np.unique(y)
        if len(labels) != 2:
            raise ValueError("Binary CSP requires exactly 2 labels.")

        C = pyriemann.utils.covariance.covariances(X)

        # Compute class means
        if self.mean_method == 'euclid':
            C0, C1 = C[y == labels[0]].mean(0), C[y == labels[1]].mean(0)
        elif self.mean_method == 'riem':
            C0 = pyriemann.utils.mean.mean_riemann(C[y == labels[0]])
            C1 = pyriemann.utils.mean.mean_riemann(C[y == labels[1]])
        else:
            raise ValueError("Invalid mean_method.")

        Csum = C0 + C1

        # Whitening transform
        eigvals, eigvecs = np.linalg.eigh(Csum)
        P = np.dot(np.diag(eigvals ** -0.5), eigvecs.T)

        # Whitened covariances and CSP eigen decomposition
        C0w = P @ C0 @ P.T
        Csw = P @ Csum @ P.T
        lmbda, V = eigh(C0w, Csw)
        idx = np.flip(np.argsort(lmbda))
        V = V[:, idx]

        # Select top/bottom eigenvectors and map back to sensor space
        Phi = np.hstack((V[:, :self.m], V[:, -self.m:]))
        W = P.T @ Phi
        A = np.abs(pinv(P.T @ V).T)

        return W, A


    def _apply_filters(self, W, X):
        """
        Apply CSP spatial filters to EEG trials.

        Parameters
        ----------
        W : ndarray
            CSP spatial filters.
            Shape:
                (n_channels, n_filters) for binary CSP, or
                (n_filter_sets, n_channels, n_filters) for multi-class extensions.
        X : ndarray, shape (n_trials, n_channels, n_samples)
            EEG data to filter.

        Returns
        -------
        X_filt : ndarray
            CSP-filtered data of shape (n_trials, n_filter_sets * n_filters, n_samples).
            For binary CSP, n_filter_sets = 1.
        """
        n_trials, n_channels, n_samples = X.shape

        # Standardize filter dimensions
        if W.ndim == 2:
            # Single-class case: expand to 3D for broadcasting
            W = W[np.newaxis, ...]  # (1, n_channels, n_filters)
        elif W.ndim != 3:
            raise ValueError("W must be 2D or 3D (n_channels, n_filters) or (n_filter_sets, n_channels, n_filters)")

        n_filt_sets, n_ch_W, n_filters = W.shape

        # Sanity check
        if n_ch_W != n_channels:
            raise ValueError(f"Channel mismatch: W has {n_ch_W}, but X has {n_channels}")

        # Vectorized application:
        #   For each filter set i and trial t:
        #       X_filt[t, i, :, :] = W[i].T @ X[t]
        #
        # einsum notation:
        #   t = trial, c = filter set, n = channels, f = filters, s = samples
        #   X_filt[t, c, f, s] = W[c, n, f]^T * X[t, n, s]
        X_filt = np.einsum('cnf,tns->tcfs', W, X)

        # Reshape to (n_trials, n_filter_sets * n_filters, n_samples)
        X_filt = X_filt.reshape(n_trials, n_filt_sets * n_filters, n_samples)

        return X_filt


    def log_var(self, X):
        """Compute log-variance CSP features."""
        var_feats = np.log(np.var(X, axis=-1, ddof=0, keepdims=False))
        return var_feats


class CSP_LDA:
    """Combined CSP and LDA pipeline."""

    def __init__(
            self, m=2, multi_class_mode='OVR',
            mean_method='euclid', log_var_feats=True
    ):
        self.csp = CSP(m, multi_class_mode, mean_method, log_var_feats)
        self.lda = LDA(solver='lsqr', shrinkage='auto')

    def fit(self, X, y):
        X_csp = self.csp.fit_transform(X, y)
        self.lda.fit(X_csp, y)
        return self

    def transform(self, X):
        return self.lda.transform(self.csp.transform(X))

    def predict(self, X):
        return self.lda.predict(self.csp.transform(X))

    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


def RWCA(X, y, cv_method='LOO', metric='accuracy', repeats=100):
    """
    Compute RWCA metric [1,2]_ using CSP+LDA classification.

    Parameters
    ----------
    X : ndarray, shape (n_trials, n_channels, n_samples)
        EEG data.
    y : ndarray, shape (n_trials,)
        Labels.
    cv_method : {'LOO', 'KFold'}, default='LOO'
        Cross-validation method. 'LOO' for leave-one-out, 'KFold' for 
        stratified k-fold (k=3).
    metric : {'accuracy', 'balanced_accuracy', 'recall', 'f1'}
        Evaluation metric.
    repeats : int, default=100
        Number of stratified CV repetitions (for KFold mode).

    Returns
    -------
    rwca : float
        Computed run-wise classification accuracy performance metric.

    References
    ----------
    .. [1] Lotte, F., & Jeunet, C. (2018). Defining and quantifying users’
           mental imagery-based BCI skills: a first step. Journal of neural 
           engineering, 15(4), 046030.

    .. [2] Ivanov, N., Wong, M., and Chau, T. (2025). A multi-class intra-trial
           trajectory analysis technique to visualize and quantify variability
           of mental imagery EEG signals. International Journal of Neural 
           Systems. https://doi.org/10.1142/S0129065725500753.

    """
    n_classes = len(np.unique(y))
    clf = CSP_LDA(m=1, log_var_feats=True)
    n_trials = len(y)

    if cv_method == 'LOO':
        y_pred = np.zeros(n_trials)
        for i in range(n_trials):
            idx_train = np.arange(n_trials) != i
            clf.fit(X[idx_train], y[idx_train])
            y_pred[i] = clf.predict(X[i:i + 1])[0]
    else:
        y_pred = np.zeros((n_trials, repeats))
        for r in range(repeats):
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=r)
            for tr_idx, te_idx in skf.split(X, y):
                clf.fit(X[tr_idx], y[tr_idx])
                y_pred[te_idx, r] = clf.predict(X[te_idx])
        y_pred = stats.mode(y_pred, axis=1)[0].squeeze()

    metrics_map = {
        'accuracy': metrics.accuracy_score,
        'balanced_accuracy': metrics.balanced_accuracy_score,
        'recall': lambda y, yp: metrics.recall_score(y, yp, average=None),
        'f1': lambda y, yp: metrics.f1_score(y, yp, average=None),
    }

    if metric not in metrics_map:
        raise ValueError(f"Invalid metric '{metric}'.")

    return metrics_map[metric](y, y_pred)
