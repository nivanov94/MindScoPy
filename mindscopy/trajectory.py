import numpy as np
from .utils.transition_matrix import count_sub_epoch_state_transitions
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from .cluster_base import Unsupervised_Segmentation


class Trajectory_Subspace(Unsupervised_Segmentation):

    def __init__(self, clustering_model=None, n_clusters=None, k_selection_thresh=0.3, krange=range(2, 12)):
        super().__init__(clustering_model, n_clusters, k_selection_thresh, krange)
        self.subspace_bases = None
        self.ref = None # the reference point that acts as the origin for the subspace


    def fit(self, X, verbose=False):
        
        # fit the clustering model
        self.fit_cluster_model(X, verbose)

        ## define the subspace spanned by the cluster centers
        # compute the mean of the cluster centers
        self.ref = np.mean(self.clustering_model.cluster_centers_, axis=0)

        # compute the subspace bases
        w = (self.clustering_model.cluster_centers_ - self.ref).T

        # compute the SVD of the cluster centers
        U, S, V = np.linalg.svd(w, full_matrices=False)

        if verbose:
            print(f'Singular values: {S}')
            print(f'w shape: {w.shape}, U shape: {U.shape}')

        # extract the subspace bases
        for i in range(len(S)):
            if S[i] < 1e-10:
                break
        
        self.subspace_bases = U[:, :i]
        self._subspace_dim = i

        return self

    
    def transform(self, X, y=None):
        
        # project the data onto the subspace
        X = np.reshape(X, (X.shape[0]*X.shape[1], X.shape[2]))
        V = X - self.ref
        V = np.dot(V, self.subspace_bases)

        # generate the state sequences
        S = self.clustering_model.predict(X)

        return V, S


class Trajectory:

    plot_colors = (
        'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 
        'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
    )

    def __init__(self, subspace_mdl):
        self.subspace_mdl = subspace_mdl

        self.basis_projs = None # Nt x Ne x K-1
        self.state_sequences = None # Nt x Ne

        self.mean_basis_projs = None # Ne x K-1
        self.epoch_cov_basis_projs = None # Ne x K-1 x K-1
        self.trial_cov_basis_projs = None # K-1 x K-1


    def fit(self, X, y=None):
        Nt, Ne = X.shape[0:2]

        # compute the state sequences and sub-space projections
        V, S = self.subspace_mdl.transform(X)

        self.basis_projs = np.zeros(
            (Nt, Ne, self.subspace_mdl._subspace_dim)
        )
        self.state_sequences = np.zeros((Nt, Ne))

        # reshape the projections and state sequences to be in trial-epoch format
        V = np.reshape(V, (Nt, Ne, -1))
        S = np.reshape(S, (Nt, Ne))

        self.basis_projs = V
        self.state_sequences = S

        # compute mean and covariance of the basis projs
        self.mean_basis_projs = np.mean(self.basis_projs, axis=0)
        self.epoch_cov_basis_projs = np.zeros(
            (Ne, self.subspace_mdl._subspace_dim, self.subspace_mdl._subspace_dim)
        )
        self.total_cov_basis_projs = np.zeros(
            (self.subspace_mdl._subspace_dim, self.subspace_mdl._subspace_dim)
        )

        for i in range(Ne):
            self.epoch_cov_basis_projs[i] = LedoitWolf().fit(
                self.basis_projs[:, i, :]
            ).covariance_

        # total covariance matrix for all epochs for intra-trial var
        self.trial_cov_basis_projs = LedoitWolf().fit(
            self.basis_projs.reshape(-1, self.subspace_mdl._subspace_dim)
        ).covariance_

        return self


    def InterTaskDiff(self, traj2):
        # Compute the inter-task difference between two trajectories
        epoch_diffs = np.zeros((self.state_sequences.shape[1],))

        for i_e in range(self.state_sequences.shape[1]):
            mean_diff = self.mean_basis_projs[i_e] - traj2.mean_basis_projs[i_e]
            epoch_diffs[i_e] = (
                np.dot(mean_diff, mean_diff) / np.trace(self.epoch_cov_basis_projs[i_e] + traj2.epoch_cov_basis_projs[i_e])
            )

        return epoch_diffs
    
    def InterTrialVar(self, all_task_traj):
        epoch_vars = np.zeros((self.state_sequences.shape[1],))

        for i_e in range(self.state_sequences.shape[1]):
            epoch_vars[i_e] = (
                np.trace(self.epoch_cov_basis_projs[i_e]) / np.trace(all_task_traj.epoch_cov_basis_projs[i_e])
            )

        return epoch_vars
    
    def IntraTrialVar(self):
        Ne = self.state_sequences.shape[1]
        Nk = self.subspace_mdl._subspace_dim

        intra_trial_mean_var = 0

        for i_k in range(Nk):
            intra_trial_mean_var += np.var(self.mean_basis_projs[:, i_k])

        intra_trial_var = intra_trial_mean_var / np.trace(self.trial_cov_basis_projs)

        return intra_trial_var


    def plot(self):
        self.plot_trajectory()
        self.plot_trellis()


    def plot_trajectory(self, yrange=None):
        Nk = self.subspace_mdl._subspace_dim
        Ne = self.state_sequences.shape[1]

        fig, ax = plt.subplots(Nk, 1, figsize=(5, Nk*3))

        for i_k in range(Nk):
            for i_e in range(Ne):
                # box plots at each sub-epoch
                ax[i_k].boxplot(
                    self.basis_projs[:, i_e, i_k], 
                    positions=[i_e+1],
                    showfliers=False
                )

                # scatter plot of the observations at each sub-epoch
                ax[i_k].scatter(
                    np.ones(self.basis_projs.shape[0])*(i_e+1),
                    self.basis_projs[:, i_e, i_k],
                    c=self.plot_colors[i_k % len(self.plot_colors)],
                    alpha=0.5
                )

            # plot connections between observations within the same trials
            for i_t in range(self.basis_projs.shape[0]):
                ax[i_k].plot(
                    np.arange(1, Ne+1),
                    self.basis_projs[i_t, :, i_k],
                    c='black',
                    alpha=0.2
                )

            # plot the mean of the basis projections
            ax[i_k].plot(
                np.arange(1, Ne+1),
                self.mean_basis_projs[:, i_k],
                c=self.plot_colors[i_k % len(self.plot_colors)],
                linewidth=3,
                marker='o',
                markersize=10
            )

            if yrange is not None:
                ax[i_k].set_ylim(yrange)

            if i_k == Nk-1:
                ax[i_k].set_xlabel('Sub-epoch')
                ax[i_k].set_xticks(np.arange(Ne)+1)
                ax[i_k].set_xticklabels(np.arange(Ne)+1)
            else:
                ax[i_k].set_xticks([])

            ax[i_k].set_ylabel(f'Basis dim $u_{i_k}$ projections')

        plt.show()

        
    def plot_trellis(self):
        Nstates = self.subspace_mdl.clustering_model.n_clusters
        Ne = self.state_sequences.shape[1]

        transitions = count_sub_epoch_state_transitions(
            self.state_sequences, self.subspace_mdl.clustering_model.n_clusters
        )
        transitions /= np.sum(transitions[0])

        fig, ax = plt.subplots(1, 1, figsize=(5, 3))

        for i_e in range(Ne-1):
            for i_origin in range(Nstates):
                for i_destination in range(Nstates):
                    if transitions[i_e, i_origin, i_destination] > 0:
                        transition_freq = transitions[i_e, i_origin, i_destination]
                        ax.plot(
                            [i_e, i_e+1],
                            [i_origin, i_destination],
                            c=f'C{i_origin}',
                            alpha=min(1, 0.2+transition_freq),
                            linewidth=5*transition_freq
                        )

                if np.sum(transitions[i_e, i_origin, :]) > 0: 
                    state_freq = np.sum(transitions[i_e, i_origin, :])
                    ax.scatter(
                        i_e, i_origin,
                        c=f'C{i_origin}',
                        s=250*state_freq,
                        alpha=state_freq
                    )
        
        for i_destination in range(Nstates):
            if np.sum(transitions[-1, :, i_destination]) > 0:
                state_freq = np.sum(transitions[-1, :, i_destination])
                ax.scatter(
                    Ne-1, i_destination,
                    c=f'C{i_destination}',
                    s=250*state_freq,
                    alpha=state_freq
                )

        ax.set_xlabel('Sub-epoch')
        ax.set_ylabel('Pattern state')

        ax.set_xticks(np.arange(Ne))
        ax.set_xticklabels(np.arange(Ne)+1)
        ax.set_yticks(np.arange(Nstates))
        ax.set_yticklabels([f'S{i+1}' for i in range(Nstates)])

        plt.show()



    