import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import pyriemann

class ScaledTangentSpace:
    """
    Extract Riemannian tangent space features from the signal X as described in
    [1]_.
    

    References
    ----------
    .. [1] Ivanov. N, Lio, A., and Chau. T. (2023). Towards user-centric
            BCI design: Markov chain-based user assessment for mental-imagery EEG-BCIs.
            Journal of Neural Engineering, 20(6).
    """
    def __init__(self):
        self.ts = pyriemann.tangentspace.TangentSpace()
        self.scalar = StandardScaler()
        self.pipeline = make_pipeline(self.ts, self.scalar)


    def fit(self, X, y=None):
        """
        Fit the feature extractor to the signal X.

        Parameters
        ----------
        X : array_like (n_trials, n_channels, n_channels)
            The input signal to extract features from, where n_trials is the number of
            trials or epochs and n_channels is the number of channels.
        y : array_like (n_trials,), optional
            Unused parameter. Default is None.

        Returns
        -------
        self : object
            Returns the instance itself
        """
        self.pipeline.fit(X)
        return self
    
    def transform(self, X):
        """
        Transform the signal X into the Riemannian tangent space.

        Parameters
        ----------
        X : array_like (n_trials, n_channels, n_channels)
            The input signal to extract features from, where n_trials is the number of
            trials or epochs and n_channels is the number of channels.

        Returns
        -------
        X_ts : array_like (n_trials, n_channels*(n_channels+1)//2)
            The signal transformed into the Riemannian tangent space.
        """
        return self.pipeline.transform(X)
    
    def fit_transform(self, X, y=None):
        """
        Fit the feature extractor to the signal X and transform it into the
        Riemannian tangent space.

        Parameters
        ----------
        X : array_like (n_trials, n_channels, n_channels)
            The input signal to extract features from, where n_trials is the number of
            trials or epochs and n_channels is the number of channels.
        y : array_like (n_trials,), optional
            Unused parameter. Default is None.

        Returns
        -------
        X_ts : array_like (n_trials, n_channels*(n_channels+1)//2)
            The signal transformed into the Riemannian tangent space.
        """
        self.pipeline.fit(X)
        return self.pipeline.transform(X)