import numpy as np
import sklearn.model_selection


def compute_pred_strength(k, tr_mod, te_mod, X):
    """ 
    Compute the prediction strength metric for a given k [1]_.

    Parameters
    ----------
    k : int
        The number of clusters.
    tr_mod : sklearn clustering model
        The clustering model fitted on the training data.
    te_mod : sklearn clustering model
        The clustering model fitted on the testing data.
    X : array_like (n_samples, n_features)
        The input data to compute the prediction strength on.
    
    Returns
    -------
    ps : float
        The prediction strength value.
    
    References
    ----------
    .. [1] Tibshirani, R., Walther, G. (2005). Cluster Validation by Prediction
           Strength. Journal of Computational and Graphical Statistics, 14(3), 511-528.
    """
    yte = te_mod.predict(X)
    ytr = tr_mod.predict(X)
    
    Ak = np.stack([yte==ki for ki in range(k)])
    nk = np.sum(Ak, axis=1)
    ps = np.inf
    
    tot = 0
    for i_k in range(k):
        ytr_i = ytr[Ak[i_k]]
        
        s = 0
        for i, yi in enumerate(ytr_i):
            for j, yj in enumerate(ytr_i):
                if i != j and yi == yj:
                    s += 1
        
        if s != 0:
            tot += s
            s = s / (nk[i_k] * (nk[i_k]-1))

        if s < ps:
            ps = s

    return ps

def cluster_pred_strength(X, y=None, krange=range(2, 12), n_repeats=25):
    """ 
    perform the prediction strength of clustering method [1]_
    for k selection

    Parameters
    ----------
    X : array_like (n_samples, n_features)
        The input data to compute the prediction strength on.
    y : array_like (n_samples,)
        The true labels for the input data.
    krange : iterable
        The range of k values to evaluate.
    n_repeats : int
        The number of times to repeat the evaluation.

    Returns
    -------
    crit : array_like (len(krange),)
        The criterion values for each k.

    References
    ----------
    .. [1] Tibshirani, R., Walther, G. (2005). Cluster Validation by Prediction
           Strength. Journal of Computational and Graphical Statistics, 14(3), 511-528.
    """
    n_folds = 2
    ps = np.ones((len(krange), n_repeats))

    for i_r in range(n_repeats):
        if y is None:
            kf = sklearn.model_selection.KFold(
                n_splits=n_folds, shuffle=True, random_state=i_r
            )
        else:
            kf = sklearn.model_selection.StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=i_r
            )

        tr_index, te_index = next(kf.split(X, y))
        Xtr, Xte = X[tr_index], X[te_index]

        for i_k, k in enumerate(krange):
            tr_mod = sklearn.cluster.KMeans(n_clusters=k, random_state=i_r)
            te_mod = sklearn.base.clone(tr_mod)
            tr_mod.fit(Xtr)
            te_mod.fit(Xte)

            ps[i_k, i_r] = compute_pred_strength(k, tr_mod, te_mod, Xte)

    crit = np.mean(ps, axis=1) + np.std(ps, axis=1)/np.sqrt(n_repeats)

    return crit
