import numpy as np
from .utils.transition_matrix import count_sub_epoch_state_transitions
from sklearn.covariance import LedoitWolf
import matplotlib.pyplot as plt
from .cluster_base import UnsupervisedSegmentation


class SubspaceTrajectoryModel(UnsupervisedSegmentation):
    """
    Learn and represent a low-dimensional state subspace from EEG state centroids [1]_.

    This class performs SVD-based dimensionality reduction on state centroids to define a
    subspace representing the dominant EEG state dynamics. It supports projection of new
    data into this subspace and visualization of component trajectories.


    Attributes
    ----------
    ref : ndarray, shape (N,)
        Mean reference vector (centroid mean).
    subspace_bases : ndarray, shape (N, d)
        Orthonormal basis vectors defining the learned subspace.
    subspace_dim : int
        Number of retained subspace dimensions.

    References
    ----------
    .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
            trajectory analysis technique to visualize and quantify variability
            of mental imagery EEG signals. International Journal of Neural Systems. 
            doi: 10.1142/S0129065725500753.
    """

    def __init__(self, clustering_model=None, n_clusters=None, 
                 k_selection_thresh=0.3, krange=range(2, 12), prefit=False
    ):
        super().__init__(clustering_model, n_clusters, k_selection_thresh, krange, prefit)
        self.subspace_bases = None
        self.ref = None # the reference point that acts as the origin for the subspace


    def fit(self, centroids, y=None, verbose=False):
        """
        Fit a low-dimensional subspace from cluster centroids.

        Parameters
        ----------
        centroids : ndarray, shape (K, N)
            Array of K N-dimensional EEG state centroids.
        """

        # fit the clustering model
        self.fit_cluster_model(centroids, y, verbose)

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
        tol = 1e-10 * S[0]
        valid = np.sum(S > tol)
        self.subspace_bases = U[:, :valid]
        self.subspace_dim = valid

        return self

    
    def transform(self, X, y=None):
        """
        Project data into the learned subspace.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_epochs, n_feats)
            Input data to be projected, where n_trials is the number of trials,
            n_epochs is the number of epochs, and n_feats is the number of features.

        Returns
        -------
        V : ndarray, shape (n_trials, n_epochs, d)
            Low-dimensional projections.
        """
        if self.subspace_bases is None:
            raise ValueError("The model has not been fitted yet.")

        if len(X.shape) != 3:
            raise ValueError("Input data X must be 3-dimensional (n_trials, n_epochs, n_feats).")
        
        # project the data onto the subspace
        V = (X - self.ref) @ self.subspace_bases

        # generate the state sequences
        X_flat = X.reshape(-1, X.shape[2])
        S = self.clustering_model.predict(X_flat)
        S = S.reshape(X.shape[0], X.shape[1])

        return V, S


class Trajectory:
    """
    Analyze and visualize trial trajectories in the learned subspace [1].

    This class represents EEG trial trajectories as low-dimensional temporal evolutions
    within the subspace learned by `SubspaceTrajectoryModel`. It supports computing covariance-
    based variability metrics and plotting per-epoch or per-trial trajectories.


    Attributes
    ----------
    subspace_mdl : SubspaceTrajectoryModel
        Fitted subspace model.
    basis_projs : ndarray
        Projected trial trajectories (n_trials, n_epochs, n_dims).
    mean_basis_projs : ndarray
        Mean trajectory across trials.
    trial_cov_basis_projs : ndarray
        Covariance matrix of trajectory projections.

    References
    ----------
    .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
            trajectory analysis technique to visualize and quantify variability
            of mental imagery EEG signals. International Journal of Neural Systems. 
            doi: 10.1142/S0129065725500753.
    """

    def __init__(self, subspace_mdl):
        self.subspace_mdl = subspace_mdl

        self.basis_projs = None # n_trials x n_epochs x K-1
        self.state_sequences = None # n_trials x n_epochs

        self.mean_basis_projs = None # n_epochs x K-1
        self.epoch_cov_basis_projs = None # n_epochs x K-1 x K-1
        self.trial_cov_basis_projs = None # K-1 x K-1


    def fit(self, X, y=None):
        """
        Compute trial projections and subspace-level covariances.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_epochs, n_features)
            EEG feature matrices for each trial.

        Returns
        -------
        self : Trajectory
            Fitted trajectory object.
        """
        n_trials, n_epochs = X.shape[0:2]

        # compute the state sequences and sub-space projections
        self.basis_projs, self.state_sequences = self.subspace_mdl.transform(X)

        # compute mean and covariance of the basis projs
        self.mean_basis_projs = np.mean(self.basis_projs, axis=0)
        self.epoch_cov_basis_projs = np.zeros(
            (n_epochs, self.subspace_mdl.subspace_dim, self.subspace_mdl.subspace_dim)
        )
        self.total_cov_basis_projs = np.zeros(
            (self.subspace_mdl.subspace_dim, self.subspace_mdl.subspace_dim)
        )

        for i in range(n_epochs):
            self.epoch_cov_basis_projs[i] = LedoitWolf().fit(
                self.basis_projs[:, i, :]
            ).covariance_

        # total covariance matrix for all epochs for intra-trial var
        self.trial_cov_basis_projs = LedoitWolf().fit(
            self.basis_projs.reshape(-1, self.subspace_mdl.subspace_dim)
        ).covariance_

        return self


    def inter_task_diff(self, other_traj):
        """
        Compute the inter-task difference between this trajectory and another
        trajectory [1].

        Parameters
        ----------
        other : Trajectory
            Another trajectory object to compare.

        Returns
        -------
        float
            Inter-task difference metric.

        References
        ----------
        .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
                trajectory analysis technique to visualize and quantify variability
                of mental imagery EEG signals. International Journal of Neural Systems. 
                doi: 10.1142/S0129065725500753.
        """
        # Compute the inter-task difference between two trajectories
        epoch_diffs = np.zeros((self.state_sequences.shape[1],))

        for i_e in range(self.state_sequences.shape[1]):
            mean_diff = self.mean_basis_projs[i_e] - other_traj.mean_basis_projs[i_e]
            epoch_diffs[i_e] = (
                np.dot(mean_diff, mean_diff) / np.trace(self.epoch_cov_basis_projs[i_e] + other_traj.epoch_cov_basis_projs[i_e])
            )

        return epoch_diffs
    
    def inter_trial_var(self, all_task_traj):
        """
        Compute the inter-trial variance of the trajectory relative to all tasks
        [1].

        Parameters
        ----------
        all_task_traj : Trajectory
            The trajectory object containing all tasks for comparison.

        Returns
        -------
        ndarray
            Inter-trial variance metric.

        References
        ----------
        .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
                trajectory analysis technique to visualize and quantify variability
                of mental imagery EEG signals. International Journal of Neural Systems. 
                doi: 10.1142/S0129065725500753.
        """
        epoch_vars = np.zeros((self.state_sequences.shape[1],))

        for i_e in range(self.state_sequences.shape[1]):
            epoch_vars[i_e] = (
                np.trace(self.epoch_cov_basis_projs[i_e]) / np.trace(all_task_traj.epoch_cov_basis_projs[i_e])
            )

        return epoch_vars


    def intra_trial_var(self):
        """
        Compute the intra-trial variance of the trajectory [1].

        Returns
        -------
        float
            Intra-trial variance metric.

        References
        ----------
        .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
                trajectory analysis technique to visualize and quantify variability
                of mental imagery EEG signals. International Journal of Neural Systems. 
                doi: 10.1142/S0129065725500753.
        """
        n_epochs = self.state_sequences.shape[1]
        Nk = self.subspace_mdl.subspace_dim

        intra_trial_mean_var = 0

        for i_k in range(Nk):
            intra_trial_mean_var += np.var(self.mean_basis_projs[:, i_k])

        intra_trial_var = intra_trial_mean_var / np.trace(self.trial_cov_basis_projs)

        return intra_trial_var


    def plot_trajectory(self, yrange=None, ax=None, plot_all_pts=False, box_plot=False, plot_colors=None):
        """
        Plot the trajectory of the trial projections over epochs.

        Parameters
        ----------
        yrange : tuple, optional
            Y-axis limits for the plots. Default is None.

        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created.
            Default is None.
        plot_all_pts : bool, optional
            Whether to plot all individual trial points. Default is False.
        box_plot : bool, optional
            Whether to use box plots for the distributions. Default is False.
        plot_colors : list of str, optional
            Colors to use for plotting each subspace dimension. Default uses
            Matplotlib tab colors.
        """
        Nk = self.subspace_mdl.subspace_dim
        n_epochs = self.state_sequences.shape[1]
        fig, ax, show = _setup_axes(Nk, 1, figsize=(5, Nk * 3), ax=ax)
        if Nk == 1:
            ax = [ax]

        if plot_colors is None:
            plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

        for i_k in range(Nk):
            color = plot_colors[i_k % len(plot_colors)]
            data = self.basis_projs[:, :, i_k]
            _plot_trajectory_dim(ax[i_k], data, color, box_plot, plot_all_pts)

            if yrange is not None:
                ax[i_k].set_ylim(yrange)
            ax[i_k].set_ylabel(f"$u_{i_k}$")
            if i_k == Nk - 1:
                ax[i_k].set_xlabel("Epoch")
                ax[i_k].set_xticks(np.arange(1, n_epochs + 1))

        if show:
            plt.show()

        
    def plot_trellis(self, ax=None, plot_colors=None):
        """
        Plot the trellis diagram of state transitions over epochs [1].

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
            Axes object to plot on. If None, a new figure and axes are created.
            Default is None.

        plot_colors : list of str, optional
            Colors to use for states. Default uses Matplotlib tab colors.

        References
        ----------
        .. [1] Ivanov, N, Wong, M., and Chau, T. (2025). A multi-class intra-trial
                trajectory analysis technique to visualize and quantify variability
                of mental imagery EEG signals. International Journal of Neural Systems. 
                doi: 10.1142/S0129065725500753.
        """
        fig, ax, show = _setup_axes(1, 1, figsize=(5, 3), ax=ax)
        if isinstance(ax, np.ndarray):
            ax = ax[0]

        _plot_trellis(
            ax, self.state_sequences, 
            self.subspace_mdl.clustering_model.n_clusters,
            plot_colors=plot_colors
        )

        if show:
            plt.show()


def plot_trajectories(trajectories, plot_all_pts=False, box_plot=False):
    """
    Plot the trajectories for multiple tasks side by side.

    Parameters
    ----------
    trajectories : list of Trajectory
        List of trajectory objects for different tasks.
    plot_all_pts : bool, optional
        Whether to plot all individual trial points. Default is False.
    box_plot : bool, optional
        Whether to use box plots for the distributions. Default is False.
    """
    n_tasks = len(trajectories)
    n_dims = trajectories[0].subspace_mdl.subspace_dim
    fig, ax = plt.subplots(n_dims, n_tasks, figsize=(n_tasks * 5, n_dims * 3))
    if n_dims == 1:
        ax = np.atleast_2d(ax)

    plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for i_task, traj in enumerate(trajectories):
        for i_dim in range(n_dims):
            color = plot_colors[i_dim % len(plot_colors)]
            data = traj.basis_projs[:, :, i_dim]
            _plot_trajectory_dim(ax[i_dim, i_task], data, color, box_plot, plot_all_pts)
            if i_dim == 0:
                ax[i_dim, i_task].set_title(f"Task {i_task + 1}")
            if i_task == 0:
                ax[i_dim, i_task].set_ylabel(f"$u_{i_dim}$")

        ax[-1, i_task].set_xlabel("Epoch")

    plt.tight_layout()


def plot_trellises(trajectories):
    """
    Plot the trellis diagrams for multiple tasks side by side.

    Parameters
    ----------
    trajectories : list of Trajectory
        List of trajectory objects for different tasks.
    """
    n_tasks = len(trajectories)
    fig, ax, _ = _setup_axes(1, n_tasks, figsize=(5 * n_tasks, 3))

    if n_tasks == 1:
        ax = [ax]

    for i_task, traj in enumerate(trajectories):
        traj.plot_trellis(ax=ax[i_task])
        ax[i_task].set_title(f"Task {i_task + 1}")

    plt.tight_layout()
    plt.show()


def _setup_axes(nrows=1, ncols=1, figsize=(5, 3), ax=None):
    if ax is None:
        fig, ax = plt.subplots(nrows, ncols, figsize=figsize)
        show = True
    else:
        fig = ax.figure
        show = False
    return fig, ax, show


def _plot_trellis(ax, state_sequences, n_states, plot_colors=None):
    """
    Plot a single trellis diagram of state transitions over epochs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    state_sequences : array_like (n_trials, n_epochs)
        The trial state sequences.
    n_states : int
        Number of discrete pattern states.
    plot_colors : list of str, optional
        Colors to use for states. Default uses Matplotlib tab colors.
    """
    from .utils.transition_matrix import count_sub_epoch_state_transitions

    if plot_colors is None:
        plot_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_epochs = state_sequences.shape[1]
    n_trials = state_sequences.shape[0]

    transitions = count_sub_epoch_state_transitions(state_sequences, n_states)
    transitions /= np.sum(transitions[0])

    for i_e in range(n_epochs - 1):
        for i_origin in range(n_states):
            for i_dest in range(n_states):
                freq = transitions[i_e, i_origin, i_dest]
                if freq > 0:
                    ax.plot([i_e, i_e + 1], [i_origin, i_dest],
                            c=plot_colors[i_origin % len(plot_colors)],
                            alpha=min(1, 0.2 + freq),
                            linewidth=5 * freq)

            state_freq = np.sum(transitions[i_e, i_origin, :])
            if state_freq > 0:
                ax.scatter(i_e, i_origin,
                           c=plot_colors[i_origin % len(plot_colors)],
                           s=250 * state_freq,
                           alpha=state_freq)

    # plot last epoch states
    for i_dest in range(n_states):
        state_freq = np.sum(transitions[-1, :, i_dest])
        if state_freq > 0:
            ax.scatter(n_epochs - 1, i_dest,
                       c=plot_colors[i_dest % len(plot_colors)],
                       s=250 * state_freq,
                       alpha=state_freq)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pattern state")
    ax.set_xticks(np.arange(n_epochs))
    ax.set_xticklabels(np.arange(1, n_epochs + 1))
    ax.set_yticks(np.arange(n_states))
    ax.set_yticklabels([f"S{i+1}" for i in range(n_states)])


def _plot_trajectory_dim(ax, data, color, box_plot=False, plot_all_pts=False):
    """
    Plot a single dimension of the trajectory.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axis to plot on.
    data : ndarray, shape (n_trials, n_epochs)
        The trajectory data for a single dimension.
    color : str
        Color for the plot.
    box_plot : bool
        Whether to use box plots for the distributions. Default is False.
    plot_all_pts : bool
        Whether to plot all individual trial points. Default is False.
    """
    n_epochs = data.shape[1]
    epochs = np.arange(1, n_epochs + 1)
    mean = np.mean(data, axis=0)

    if box_plot:
        ax.boxplot(data, positions=epochs, showfliers=False)
    else:
        stds = np.std(data, axis=0)
        ax.fill_between(epochs, mean - stds, mean + stds, color=color, alpha=0.3)

    if plot_all_pts:
        for i_e in range(n_epochs):
            ax.scatter(np.full(data.shape[0], i_e + 1), data[:, i_e], c=color, alpha=0.5)
        ax.plot(epochs, data.T, c='black', alpha=0.2)

    ax.plot(epochs, mean, c=color, linewidth=3, marker='o', markersize=8)