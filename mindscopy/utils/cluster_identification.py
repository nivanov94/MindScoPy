import numpy as np
import sklearn.model_selection


def compute_pred_strength(k, tr_mod, te_mod, X):
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

def cluster_pred_strength(X, y, thresh=None, krange=range(2, 12), Nrepeats=25):
    """ 
    perform the prediction strength of clustering method
    for k selection
    """

    Nfolds = 2
    ps = np.ones((len(krange), Nfolds*Nrepeats))

    for i_r in range(Nrepeats):
        skf = sklearn.model_selection.StratifiedKFold(n_splits=2, shuffle=True, random_state=i_r)

        for i_f, (tr_index, te_index) in enumerate(skf.split(X, y)):
            Xtr, Xte = X[tr_index], X[te_index]

            for i_k, k in enumerate(krange):
                tr_mod = sklearn.pipeline.Pipeline([
                    ('scale', sklearn.preprocessing.StandardScaler()),
                    ('kmeans', sklearn.cluster.KMeans(n_clusters=k, random_state=i_r))
                ])
                te_mod = sklearn.base.clone(tr_mod)
                tr_mod.fit(Xtr)
                te_mod.fit(Xte)

                ps[i_k, i_r*Nfolds+i_f] = compute_pred_strength(k, tr_mod, te_mod, Xte)


    crit = np.mean(ps, axis=1) + np.std(ps, axis=1)/np.sqrt(Nfolds*Nrepeats)

    return crit