import numpy as np
import pyriemann

def rebias_covariances(C, P):
    """
    Apply the rebiasing method from [1]_ to the covariance matrices C using the
    projection matrix P. The rebiasing method is defined as: P^(-1/2) * C * P^(-1/2).

    Parameters
    ----------
    C : array_like (Nt, Nc, Nc)
        The input covariance matrices to be rebias, where Nc is the number of
        channels and Nt is the number of trials.

    P : array_like (Nc, Nc)
        The projection matrix used to rebias the covariance matrices.

    Returns
    -------
    C_rebiased : array_like (Nt, Nc, Nc)
        The rebias covariance matrices.

    References
    ----------
    .. [1] Benaroch, C. et al. (2021). Long-Term BCI Training of a 
              Tetraplegic User: Adaptive Riemaiian Classifiers and User
              Training. Frontiers in Human Neuroscience, 15.
    """

    if P.shape[0] != P.shape[1]:
        raise ValueError("Matrix P must be square.")

    if C.ndim != 3 or C.shape[1] != P.shape[0] or C.shape[2] != P.shape[1]:
        raise ValueError("Matrix C must have shape (m, n, n) where n matches P.")

    # Eigen decomposition of C
    eigvals, eigvecs = np.linalg.eigh(P)
    
    # Check for positive definiteness
    if np.any(eigvals <= 0):
        raise np.linalg.LinAlgError("Matrix P must be positive definite.")

    # Compute C^(-1/2)
    P_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

    # Efficient batch transformation using np.einsum
    return np.einsum("ij,mjk,kl->mil", P_inv_sqrt, C, P_inv_sqrt)


def apply_rebias_to_groups(X, group_labels):
    """
    Applies the rebiasing method to the covariance matrices in X based on the
    group labels provided in group_labels. The mean of each group is calculated
    and then used to compute the projection matrix P. The rebiasing method is
    defined as: P^(-1/2) * C * P^(-1/2).

    Parameters
    ----------
    X : array_like (Nt, Nc, Nc)
        The input covariance matrices to be rebias, where Nc is the number of
        channels and Nt is the number of trials.
    
    group_labels : array_like (Nt,)
        The group labels for each trial in X.

    Returns
    -------
    X_rebiased : array_like (Nt, Nc, Nc)
        The rebias covariance matrices.
    """
    
    unique_groups = np.unique(group_labels)
    X_rebiased = np.zeros_like(X)

    for group in unique_groups:
        group_indices = np.where(group_labels == group)[0]
        group_covs = X[group_indices]
        group_mean = pyriemann.utils.mean.mean_covariance(group_covs)

        X_rebiased[group_indices] = rebias_covariances(group_covs, group_mean)

    return X_rebiased