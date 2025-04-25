import mindscopy as ms
from mindscopy.preprocessing.artifact_removal import peak_rejection, riemannian_potato_rejection
from mindscopy.preprocessing.misc import epoch
from mindscopy.preprocessing.rebias import apply_rebias_to_groups
from mindscopy.preprocessing.feature_extraction import ScaledTangentSpace
import numpy as np
import pyriemann
import pickle
from matplotlib import pyplot as plt

# load the preprocessed (bandpass filtered and epoched) BCI Competition IV 2a data
with open('data/BCI_Comp_IV_2a/preprocessed_data.pkl', 'rb') as f:
    d = pickle.load(f)

# Some parameters
Fs = 250
krange = range(3, 10)
Nblks = 12
tasks = (1, 2, 3, 4)
Ntasks = len(tasks)

chs = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_map = {
    ch : i for i, ch in enumerate(chs)
}
Nc = len(chs)

# allocate metric results
Np = 9
inter_task_diff = np.zeros((Np, Nblks, Ntasks))
inter_trial_var = np.zeros((Np, Nblks, Ntasks))
intra_trial_var = np.zeros((Np, Nblks, Ntasks))
rwca = np.zeros((Np, Nblks))

# iterate over participants
for i_p, p in enumerate(range(1,Np+1)):
    print("*"*80)
    print(f"Participant {p}...")

    # Extract all data from both sessions
    X = np.concatenate([d[p][1]['trials'], d[p][2]['trials']])
    y = np.concatenate([d[p][1]['labels'], d[p][2]['labels']])
    blocks = np.concatenate([
        d[p][1]['run_labels'], 
        d[p][2]['run_labels']+(1+max(d[p][1]['run_labels'])) # shift the second session's block labels
    ])

    # remove artifacts
    # first apply peak rejection
    X, rejected_trials = peak_rejection(X, threshold=350, verbose=True)
    y = np.delete(y, rejected_trials)
    blocks = np.delete(blocks, rejected_trials)

    # then apply riemannian potato rejection
    clean_X = []
    clean_y = []
    clean_blocks = []
    for block in np.unique(blocks):
        print(f"Applying Riemannian potato rejection to block {block+1}")
        block_idx = blocks == block
        X_block = X[block_idx]
        y_block = y[block_idx]
        blocks_block = blocks[block_idx]

        X_block, rejected_trials = riemannian_potato_rejection(X_block, threshold=2.5, verbose=True)
        y_block = np.delete(y_block, rejected_trials)
        blocks_block = np.delete(blocks_block, rejected_trials)

        clean_X.append(X_block)
        clean_y.append(y_block)
        clean_blocks.append(blocks_block)

    X = np.concatenate(clean_X)
    y = np.concatenate(clean_y)
    blocks = np.concatenate(clean_blocks)

    print("Preprocessing ...")
    # apply rebias to each block to reduce non-stationarity between blocks/sessions
    Nt, Nc, Ns = X.shape
    Nblks = len(np.unique(blocks))

    # block means
    block_means = np.zeros((Nblks, Nc, Nc))
    for i, block in enumerate(np.sort(np.unique(blocks))):
        block_covs = pyriemann.utils.covariance.covariances(X[blocks == block])
        block_means[i] = pyriemann.utils.mean.mean_covariance(block_covs)

    # Generate sub-epochs for each trial
    length = 2.0
    stride_traj = 0.5
    stride_clust = 1.0
    Ns_epoch = int(Fs * length)
    X_clust = epoch(X, Ns_epoch, int(Fs * stride_clust))
    X_traj = epoch(X, Ns_epoch, int(Fs * stride_traj))
    N_clust_epochs = X_clust.shape[1]
    N_traj_epochs = X_traj.shape[1]

    # apply the rebiasing to the sub-epochs
    blocks_clust_epochs = np.repeat(blocks, N_clust_epochs)
    blocks_traj_epochs = np.repeat(blocks, N_traj_epochs)

    X_clust = np.reshape(X_clust, (-1, Nc, Ns_epoch))
    X_traj = np.reshape(X_traj, (-1, Nc, Ns_epoch))

    X_clust_covs = pyriemann.utils.covariance.covariances(X_clust)
    X_traj_covs = pyriemann.utils.covariance.covariances(X_traj)
    
    X_clust_covs = apply_rebias_to_groups(
        X_clust_covs, blocks_clust_epochs, block_means
    )
    X_traj_covs = apply_rebias_to_groups(
        X_traj_covs, blocks_traj_epochs, block_means
    )

    # apply feature extraction
    feat_extrator = ScaledTangentSpace().fit(X_clust_covs)
    X_clust_feats = feat_extrator.transform(X_clust_covs)
    X_traj_feats = feat_extrator.transform(X_traj_covs)

    # put the data back into (trial, epoch, feature) format
    X_clust_feats = np.reshape(X_clust_feats, (Nt, N_clust_epochs, -1))
    X_traj_feats = np.reshape(X_traj_feats, (Nt, N_traj_epochs, -1))

    print("Generating subspace...")
    # Perform clustering and generate the trajectory subspace
    np.random.seed(42)
    Traj_subspace = ms.Trajectory_Subspace(krange=krange).fit(
        X_clust_feats, y=y
    )
    print(f"Subspace dimension: {Traj_subspace._subspace_dim}")
    
    # compute the metrics for each block and task
    print("Computing metrics...")
    for i_blk, blk in enumerate(range(Nblks)):
        print(f"\tBlock {blk}...")

        # get the block data - This is the full trial data for classification
        X_blk = X[blocks == blk]
        y_blk = y[blocks == blk]

        # get the Epoch features for the block
        X_feats_blk = X_traj_feats[blocks == blk]

        # compute the trajectory -- TODO from here...
        Traj_all_tasks = ms.Trajectory(Traj_subspace).fit(X_feats_blk, y=y_block)

        for i_t, task in enumerate(tasks):
            print(f"\t\tTask {task}...")

            # get the task data
            X_feats_task = X_feats_blk[y_blk == task]
            y_task = y_blk[y_blk == task]

            # compute the trajectory
            Traj_task = ms.Trajectory(Traj_subspace).fit(X_feats_task, y=y_task)

            # compute the InterTaskDiff
            itd = Traj_task.InterTaskDiff(Traj_all_tasks)
            inter_task_diff[i_p, i_blk, i_t] = np.mean(itd) # task the mean over all epochs for correlation analysis

            # compute the InterTrialVar
            itv = Traj_task.InterTrialVar(Traj_all_tasks)
            inter_trial_var[i_p, i_blk, i_t] = np.mean(itv)

            # compute the IntraTrialVar
            intra_trial_var[i_p, i_blk, i_t] = Traj_task.IntraTrialVar()

        # compute the RWCA
        # use the mean because the RWCA returns F1 scores for each task individuallly
        rwca[i_p, i_blk] = np.mean(ms.RWCA(X_blk, y_blk, cv_method='k-fold', metric='f1', repeats=100))


# plot the results
f, ax = plt.subplots(1, 2, figsize=(10, 5))

colours = ('tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan')

# plt InterTaskDiff vs. RWCA
for i_p in range(Np):
    for i_blk in range(Nblks):
        # use the mean of the inter-task differences across the tasks
        ax[0].scatter(
            np.log(np.mean(inter_task_diff[i_p, i_blk, :])), rwca[i_p, i_blk],
            label=f'P{i_p+1}' if i_blk == 0 else None,
            color=colours[i_p]
        )

    # add the mean for the participant
    ax[0].scatter(
        np.log(np.mean(np.mean(inter_task_diff[i_p], axis=1))), np.mean(rwca[i_p]),
        label=f'P{i_p+1} - Mean', color=colours[i_p], marker='x'
    )

ax[0].set_ylabel('RWCA')
ax[0].set_xlabel('InterTaskDiff')
ax[0].legend()

# plt InterTrialVar vs. RWCA
for i_p in range(Np):
    for i_blk in range(Nblks):
        # use the mean of the inter-task differences across the tasks
        ax[1].scatter(
            np.mean(inter_trial_var[i_p, i_blk, :]), rwca[i_p, i_blk],
            label=f'P{i_p+1}' if i_blk == 0 else None,
            color=colours[i_p]
        )

    # add the mean for the participant
    ax[1].scatter(
        np.mean(np.mean(inter_trial_var[i_p], axis=1)), np.mean(rwca[i_p]),
        label=f'P{i_p+1}', color=colours[i_p], marker='x'
    )

ax[1].set_ylabel('RWCA')
ax[1].set_xlabel('InterTrialVar')

plt.show()

# plot the InterTrialVar vs. IntraTrialVar
f, ax = plt.subplots(1, 1, figsize=(5, 5))

for i_p in range(Np):
    ax.scatter(
        inter_trial_var[i_p], intra_trial_var[i_p],
        label=f'P{i_p+1}', color=colours[i_p]
    )

ax.set_xlabel('InterTrialVar')
ax.set_ylabel('IntraTrialVar')
ax.legend()
plt.show()