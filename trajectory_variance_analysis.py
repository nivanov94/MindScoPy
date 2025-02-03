import mindscopy as ms
from preprocessing.artifact_removal import peak_rejection, riemannian_potato_rejection
from preprocessing.misc import epoch
from preprocessing.rebias import apply_rebias_to_groups
from preprocessing.feature_extraction import ScaledTangentSpace
import numpy as np
import pyriemann
import pickle
from sklearn.model_selection import StratifiedKFold
from mindscopy.utils.cluster_identification import cluster_pred_strength
from sklearn.metrics import f1_score
from matplotlib import pyplot as plt
from scipy import stats

# load the preprocessed (bandpass filtered and epoched) BCI Competition IV 2a data
with open('data/BCI_Comp_IV_2a/preprocessed_data.pkl', 'rb') as f:
    d = pickle.load(f)

# Some parameters
Fs = 250
krange = range(3, 10)
Nblocks = 12
Ntasks = 4

chs = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_map = {
    ch : i for i, ch in enumerate(chs)
}
Nc = len(chs)

# allocate metric results
Np = 9
inter_task_diff = np.zeros((Np, Nblocks, Ntasks))
inter_trial_var = np.zeros((Np, Nblocks, Ntasks))
intra_trial_var = np.zeros((Np, Nblocks, Ntasks))
rwca = np.zeros((Np, Nblocks))

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

    Xcluster = np.reshape(Xcluster, (-1, Nc, Ns_epoch))
    Xtraj = np.reshape(Xtraj, (-1, Nc, Ns_epoch))

    Xclust_covs = pyriemann.utils.covariance.covariances(Xcluster)
    Xtraj_covs = pyriemann.utils.covariance.covariances(Xtraj)
    
    Xclust_covs = apply_rebias_to_groups(
        Xclust_covs, blocks_clust_epochs, block_means
    )
    Xtraj_covs = apply_rebias_to_groups(
        Xtraj_covs, blocks_traj_epochs, block_means
    )

    # apply feature extraction
    feat_extrator = ScaledTangentSpace().fit(Xclust_covs)
    Xclust_feats = feat_extrator.transform(Xclust_covs)
    Xtraj_feats = feat_extrator.transform(Xtraj_covs)

    # put the data back into (trial, epoch, feature) format
    Xclust_feats = np.reshape(Xclust_feats, (Nt, N_clust_epochs, -1))
    Xtraj_feats = np.reshape(Xtraj_feats, (Nt, N_traj_epochs, -1))

    # Perform clustering and generate the trajectory subspace
    np.random.seed(42)
    traj_clust = ms.Trajectory_Subspace(krange=krange).fit(
        Xclust_feats, y=y
    )
    
    # compute the metrics for each block and task
    for i_blk, blk in enumerate(np.unique(blocks)):
        print(f"Block {block+1}...")

        # get the block data
        X_blk = X[blocks == blk]
        y_blk = y[blocks == blk]

        # compute the trajectory -- TODO from here...
        traj = ms.Trajectory(traj_clust).fit(X_block, y=y_block)

        # compute the inter-task distinctness
        inter_task_diff[i_p, i_blk] = ms.task_distinct(traj.state_sequences)

        # compute the inter-trial variance
        inter_trial_var[i_p, i_blk] = ms.relative_task_inconsistency(traj.state_sequences)

        # compute the intra-trial variance
        intra_trial_var[i_p, i_blk] = ms.relative_task_inconsistency(traj.basis_projs)

        # compute the RWCA
        rwca[i_p, i_blk] = ms.RWCA(traj.state_sequences, y_block)
    print(f"Task distinct: {np.mean(task_distinct[i_p])}")
    print(f"Task inconsistency: {np.mean(task_incon[i_p])}")
    print(f"Classification F1 score: {np.mean(clsf_f1[i_p])}")

print(np.mean(task_distinct, axis=1))
print(np.mean(task_incon, axis=1))
print(np.mean(clsf_f1, axis=1))

# plot the results
f, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(np.mean(task_distinct, axis=1), np.mean(clsf_f1, axis=1))
ax[0].set_xlabel('taskDistinct')
ax[0].set_ylabel('F1 score')

ax[1].scatter(np.mean(task_incon, axis=1), np.mean(clsf_f1, axis=1))
ax[1].set_xlabel('relativeTaskInconsistency')
ax[1].set_ylabel('F1 score')

# write the participant numbers next to the points
for i, p in enumerate(range(1, Np+1)):
    ax[0].annotate(f'P{p}', (np.mean(task_distinct, axis=1)[i]+0.005, np.mean(clsf_f1, axis=1)[i]+0.01))
    ax[1].annotate(f'P{p}', (np.mean(task_incon, axis=1)[i]+0.005, np.mean(clsf_f1, axis=1)[i]+0.01))

# add the correlation lines
slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.mean(task_distinct, axis=1), np.mean(clsf_f1, axis=1)
)
x = np.linspace(
    min(np.mean(task_distinct, axis=1)),
    max(np.mean(task_distinct, axis=1)),
    100
)
y = slope * x + intercept
ax[0].plot(x, y, 'r--')
# write the correlation coefficient on the plot
ax[0].annotate(
    f'r={r_value:.3f}, p={p_value:.3f}',
    (0.55, 0.15), xycoords='axes fraction'
)

slope, intercept, r_value, p_value, std_err = stats.linregress(
    np.mean(task_incon, axis=1), np.mean(clsf_f1, axis=1)
)
x = np.linspace(
    min(np.mean(task_incon, axis=1)),
    max(np.mean(task_incon, axis=1)),
    100
)
y = slope * x + intercept
ax[1].plot(x, y, 'r--')
# write the correlation coefficient on the plot
ax[1].annotate(
    f'r={r_value:.3f}, p={p_value:.3f}',
    (0.05, 0.15), xycoords='axes fraction'
)

plt.show()