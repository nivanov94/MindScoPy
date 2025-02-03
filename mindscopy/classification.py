import numpy as np
from itertools import combinations as iter_combs
from scipy.linalg import eigh
import scipy.stats
import pyriemann
import mne
from matplotlib import pyplot as plt
from scipy.special import binom
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import StratifiedKFold
import sklearn.metrics

class CSP:
    """
    Vanilla CSP - no feature optimization, single freq band
    """
    
    def __init__(self,
                 classes,
                 m=2,
                 multi_class_ext='OVR',
                 mean_method='euclid',
                 log_var_feats=False):
        
        self.classes = classes
        self.multi_class_ext = multi_class_ext
        self.m = m
        self.mean_method = mean_method
        self.log_var_feats = log_var_feats
        
        self.filters_ = None
        self.patterns_ = None
    
    
    def transform(self,X):
        Xcsp = self._apply_csp_filts(self.filters_, X).squeeze()

        if self.log_var_feats:
            return self.log_var(Xcsp)
        else:
            return Xcsp
        
    def fit(self,X,y):
        # Calculate spatial filters and extract features
        W, A = self._calc_csp_filters(X,y)
        
        self.patterns_ = A
        self.filters_ = W
    
    def fit_transform(self,X,y):
        self.fit(X,y)
        return self.transform(X)
    
    def _calc_csp_filters(self,X,y):
        """
	Compute the CSP filters using the inputted data
	"""

        if self.classes == 2:
            return self._calc_binary_csp_filters(X,y)
        
        if self.multi_class_ext == 'OVR':
	    # compute set of filters for one-vs-rest or one-vs-all classification
            _, Nc, Ns = X.shape
            labels = np.unique(y)
            Nl = labels.shape[0]
            W = np.zeros((Nl,Nc,2*self.m))
            A = np.zeros((Nl,Nc,Nc))
            
            for i_l in range(Nl):
                l = labels[i_l]
                yl = np.copy(y)
                yl[y==l] = 0 # current class label aliased as 0
                yl[y!=l] = 1 # all others aliased as 1
                
                W[i_l,:,:], A[i_l,:,:] = self._calc_binary_csp_filters(X,yl)
            
            return W, A
                
        elif self.multi_class_ext == 'PW':
	    # compute set of filters for pairwise one-vs-one classification
            _, Nc, Ns = X.shape
            labels = np.unique(y)
            Nl = labels.shape[0]
            
            Nf = int(binom(Nl,2)) # number of pairs
            
            W = np.zeros((Nf,Nc,2*self.m))
            A = np.zeros((Nf,Nc,Nc))
            
            for (i,(l1,l2)) in enumerate(iter_combs(labels,2)):
                Xl1 = X[y==l1,:,:,:]
                Xl2 = X[y==l2,:,:,:]
		# create feature and label matrices using the current pair
                yl = np.concatenate((l1 * np.ones(Xl1.shape[0],),
                                     l2 * np.ones(Xl2.shape[0])),
                                    axis=0)
                Xl = np.concatenate((Xl1,Xl2),
                                    axis=0)                
                
                W[i,:,:], A[i,:,:] = self._calc_binary_csp_filters(Xl,yl)
            
            return W, A
        
        else:
            raise Exception("Invalid multi-class extention")
        
    
    def _calc_binary_csp_filters(self,X,y):
        from scipy.linalg import pinv
        _, Nc, Ns = X.shape
        
        labels = np.sort(np.unique(y))
        Nl = labels.shape[0]
        
        if Nl != 2:
            raise Exception("invalid number of labels")
        
        # step 0 - compute the covariance for each trial
        C = pyriemann.utils.covariance.covariances(X)

        if self.mean_method == 'euclid':
            C0_bar = np.mean(C[y==labels[0]], axis=0)
            C1_bar = np.mean(C[y==labels[1]], axis=0)
        elif self.mean_method == 'riem':
            C0_bar = pyriemann.utils.mean.mean_riemann(C[y==labels[0]])
            C1_bar = pyriemann.utils.mean.mean_riemann(C[y==labels[1]])

        Ctot_bar = C0_bar + C1_bar

        # step 1 - compute the whitening transform
        l, U = np.linalg.eig(Ctot_bar)
        P = np.matmul(np.diag(l ** (-1/2)), U.T) # whitening matrix
        
        # step 2 - apply the whitening transform
        C0_bar_white = np.matmul(P, np.matmul(C0_bar, P.T))
        Ctot_white = np.matmul(P, np.matmul(Ctot_bar, P.T))

        # step 3 - compute the filters
        l, V = eigh(C0_bar_white, Ctot_white)

        # sort the eigenvalues and eigenvectors
        ix = np.flip(np.argsort(l))
        l = l[ix]
        V = V[:,ix]

        # step 4 - select eigenvectors associated with highest and lowest eigenvalue
        Phi = np.concatenate((V[:,:self.m], V[:,-self.m:]), axis=1)
        
        # step 5 - rotate the filters back into the channel space
        W = np.matmul(P.T, Phi)
        A = np.abs(pinv(np.matmul(P.T, V)).T)
        
        return W, A
    
    def _apply_csp_filts(self,W,X):
        Nt,Nc,Ns = X.shape
        
        if len(W.shape) == 2:
            Ncl = 1
            Ncw, Nf = W.shape
            W = np.expand_dims(W,axis=0)
            X_filt = np.zeros((Nt,1,Nf,Ns))
        else:
            Ncl, Ncw, Nf = W.shape
            X_filt = np.zeros((Nt,Ncl,Nf,Ns))
            
        if Ncw != Nc:
            raise Exception("Channel mismatch")
        
        for i_t in range(Nt):
            for i_f in range(Ncl):
                X_filt[i_t,i_f,:,:] = np.matmul(W[i_f,:,:].T, X[i_t,:,:])
            
        return X_filt
    
    def log_var(self,X):
        """compute log variance of each CSP filter channel"""
        if len(X.shape) == 3:
            X = np.expand_dims(X,axis=1)        

        Nt, Nf, Nc, Ns = X.shape # Nc = 2m, Nf is the number of CSP filters (for >2 class problem)

        X_log_var = np.log(np.var(X,axis=3,ddof=0)) # Nt, Nf, Nc
        X_log_var = np.resize(X_log_var, (Nt,Nf*Nc))
                
        return X_log_var
    
    def visualize_patterns(self, chs):

        info = mne.create_info(ch_names=chs, sfreq=1., ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        fig, axes = plt.subplots(nrows=1,ncols=self.patterns_.shape[0]*self.m, figsize=(14, 6))
        vmax = np.max(np.abs(self.patterns_))
        
        for i_p, pattern in enumerate(self.patterns_):
            for i_m in range(self.m):
                im, cm = mne.viz.plot_topomap(pattern[:,i_m], info, vlim=(0,vmax), show=False, axes=axes[i_p*self.m+i_m], cmap='RdBu')
    
        # manually fiddle the position of colorbar
        width = 2.5*self.patterns_.shape[0]*self.m
        ax_x_start = 0.95
        ax_x_width = 0.1 / width
        ax_y_start = 0.28
        ax_y_height = 0.5
        cbar_ax = fig.add_axes([ax_x_start, ax_y_start, ax_x_width, ax_y_height])
        clb = fig.colorbar(im, cax=cbar_ax)
        clb.ax.set_title('AU',fontsize=18) # title on top of colorbar
        clb.ax.tick_params(labelsize=18)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show()


class CSP_LDA:

    def __init__(self, classes, m=2, multi_class_ext='OVR', mean_method='euclid', log_var_feats=True):
        self.classes = classes
        self.multi_class_ext = multi_class_ext

        self.csp = CSP(classes, m, multi_class_ext, mean_method, log_var_feats)
        self.lda = LDA(solver='lsqr', shrinkage='auto')

    def fit(self, X, y):
        X_csp = self.csp.fit_transform(X, y)
        self.lda.fit(X_csp, y)

    def transform(self, X):
        X_csp = self.csp.transform(X)
        return self.lda.transform(X_csp)
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    def predict(self, X):
        X_csp = self.csp.transform(X)
        return self.lda.predict(X_csp)
    
    def fit_predict(self, X, y):
        self.fit(X, y)
        return self.predict(X)


def RWCA(X, y, cv_method='LOO', metric='accuracy', repeats=100):
    """ Compute RWCA metric for each block of trials """
    classes = len(np.unique(y))
    clsf = CSP_LDA(classes=classes, m=1, log_var_feats=True)

    Ntrials = X.shape[0]
    if cv_method == 'LOO':
        # perform LOO CV
        y_pred = np.zeros((Ntrials,))
        for i_t in range(Ntrials):
            train_index = np.delete(np.arange(Ntrials), i_t)
            test_index = i_t

            clsf.fit(X[train_index], y[train_index])
            y_pred[i_t] = clsf.predict(X[test_index])[0]
    else:
        y_pred = np.zeros((Ntrials,repeats))
        for i_r in range(repeats):
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=i_r)
            for train_index, test_index in skf.split(X, y):
                clsf.fit(X[train_index], y[train_index])
                y_pred[test_index, i_r] = clsf.predict(X[test_index])
        
        # compute the mode of the predictions
        y_pred = scipy.stats.mode(y_pred, axis=1)[0].squeeze()

    # calculate the RWCA metric
    if metric == 'accuracy':
        rwca = sklearn.metrics.accuracy_score(y, y_pred)
    elif metric == 'balanced_accuracy':
        rwca = sklearn.metrics.balanced_accuracy_score(y, y_pred, adjusted=True)
    elif metric == 'recall':
        rwca = sklearn.metrics.recall_score(y, y_pred, average=None)
    else:
        rwca = sklearn.metrics.f1_score(y, y_pred, average=None)

    return rwca
