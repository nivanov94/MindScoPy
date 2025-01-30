import pyriemann
import numpy as np

def peak_rejection(X, threshold=350, verbose=False):
    """
    Identify and remove trials with large absolute value amplitude peaks 
    in the signal X.
    
    Parameters
    ----------
    X : array_like (Nt, Nc, Ns)
        The input signal to be filtered, where Nt is the number of trials,
        Nc is the number of channels and Ns is the number of samples.

    threshold : float, optional
        The threshold value for peak rejection. Default is 350.

    verbose : bool, optional
        Whether to print the number of rejected trials. Default is False.

    Returns
    -------
    X_clean : array_like (Nt, Nc, Ns)
        The cleaned signal.
    rejected_indices : array_like (Nt,)
        The indices of the rejected trials.
    """

    Nt, Nc, Ns = X.shape

    clean_indices = np.ones(Nt, dtype=bool)

    for i in range(Nt):
        if np.any(np.abs(X[i]) > threshold):
            clean_indices[i] = False

    if verbose:
        print(f'{Nt - np.sum(clean_indices)} trials rejected.')

    X_clean = X[clean_indices]

    return X_clean, ~clean_indices


def riemannian_potato_rejection(X, threshold=2.75, max_iter=5, verbose=False):
    """
    Identify and remove trials from X using the Riemannian potato rejection 
    method from [1]_. 

    Parameters
    ----------
    X : array_like (Nt, Nc, Ns)
        The input signal to be filtered, where Nt is the number of trials,
        Nc is the number of channels and Ns is the number of samples.
    
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
    X_clean : array_like (Nt, Nc, Ns)
        The signal with rejected trials removed.
    
    rejected_indices : array_like (Nt,)
        The indices of the rejected trials.
    
    References
    ----------
    .. [1] `The Riemannian Potato: an automatic and adaptive artifact detection
        method for online experiments using Riemannian geometry
        <https://hal.archives-ouvertes.fr/hal-00781701>`_
        A. Barachant, A Andreev, and M. Congedo. TOBI Workshop lV, Jan 2013,
        Sion, Switzerland. pp.19-20.
    """
    Nt, Nc, Ns = X.shape
    trials_remaining = Nt

    clean_indices = np.ones(Nt, dtype=bool)
    Xcovs = pyriemann.utils.covariance.covariances(X)
    
    # apply some regularization to the covariance matrices to avoid singularities
    r = 1e-6
    Xcovs = (1-r)*Xcovs + r*np.eye(Nc)

    # create the potato object
    potato = pyriemann.clustering.Potato(threshold=threshold)

    for i_t in range(max_iter):
        # fit the potato object with the set of clean trials
        potato.fit(Xcovs[clean_indices])

        # identify rejected trials
        rejected_indices = potato.predict(Xcovs[clean_indices])

        # update the clean indices
        clean_indices[clean_indices] = rejected_indices

        if verbose:
            print(f'Iteration {i_t}: {trials_remaining - np.sum(clean_indices)} trials rejected.')
            trials_remaining = np.sum(clean_indices)

        if not np.any(rejected_indices):
            break

    if verbose:
        print(f'Process complete, {Nt - np.sum(clean_indices)} total trials rejected.')

    X_clean = X[clean_indices]

    return X_clean