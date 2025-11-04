import pyriemann
import numpy as np

def peak_rejection(X, threshold=350, verbose=False):
    """
    Identify and remove trials with large absolute value amplitude peaks 
    in the signal X.
    
    Parameters
    ----------
    X : array_like (n_trials, n_channels, n_samples)
        The input signal to be filtered, where n_trials is the number of trials,
        n_channels is the number of channels and n_samples is the number of samples.

    threshold : float, optional
        The threshold value for peak rejection. Default is 350.

    verbose : bool, optional
        Whether to print the number of rejected trials. Default is False.

    Returns
    -------
    X_clean : array_like (n_trials, n_channels, n_samples)
        The cleaned signal.
    rejected_indices : array_like (n_trials,)
        The indices of the rejected trials.
    """

    n_trials, n_channels, n_samples = X.shape

    clean_indices = np.ones(n_trials, dtype=bool)

    for i in range(n_trials):
        if np.any(np.abs(X[i]) > threshold):
            clean_indices[i] = False

    if verbose:
        print(f'{n_trials - np.sum(clean_indices)} trials rejected.')

    X_clean = X[clean_indices]

    return X_clean, ~clean_indices


def riemannian_potato_rejection(X, threshold=2.5, verbose=False):
    """
    Identify and remove trials from X using the Riemannian potato rejection 
    method from [1]_. 

    Parameters
    ----------
    X : array_like (n_trials, n_channels, n_samples)
        The input signal to be filtered, where n_trials is the number of trials,
        n_channels is the number of channels and n_samples is the number of samples.
    
    threshold : float, optional
        The threshold value for the Riemannian potato rejection method.
        Default is 2.75.
    
    max_iter : int, optional
        The maximum number of iterations for the Riemannian potato rejection
        method. Default is 5.
    
    verbose : bool, optional
        Whether to print the number of rejected trials. Default is False.

    Returns
    -------
    X_clean : array_like (n_trials, n_channels, n_samples)
        The signal with rejected trials removed.
    
    rejected_indices : array_like (n_trials,)
        The indices of the rejected trials.
    
    References
    ----------
    .. [1] `The Riemannian Potato: an automatic and adaptive artifact detection
        method for online experiments using Riemannian geometry
        <https://hal.archives-ouvertes.fr/hal-00781701>`_
        A. Barachant, A Andreev, and M. Congedo. TOBI Workshop lV, Jan 2013,
        Sion, Switzerland. pp.19-20.
    """
    n_trials, n_channels, n_samples = X.shape

    Xcovs = pyriemann.utils.covariance.covariances(X)
    
    # apply some regularization to the covariance matrices to avoid singularities
    r = 1e-4  # TODO parameterize regularization
    Xcovs = (1-r)*Xcovs + r*np.eye(n_channels)

    # create the potato object
    potato = pyriemann.clustering.Potato(threshold=threshold)

    # fit the potato object with the set of clean trials
    potato.fit(Xcovs)

    # identify rejected trials
    clean_indices = potato.predict(Xcovs)

    # update the clean indices
    clean_indices = clean_indices.astype(bool)

    if verbose:
        print(f'Potato filtering: {n_trials - np.sum(clean_indices)} trials rejected.')

    X_clean = X[clean_indices]

    return X_clean, ~clean_indices