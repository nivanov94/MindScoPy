import mindscopy as ms
from mindscopy.preprocessing.artifact_removal import peak_rejection, riemannian_potato_rejection
from mindscopy.preprocessing.misc import epoch
from mindscopy.preprocessing.rebias import apply_rebias_to_groups
from mindscopy.preprocessing.feature_extraction import ScaledTangentSpace
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
n_folds = 2
n_repeats = 10
n_total_folds = n_folds * n_repeats
krange = range(3, 10)

chs = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
ch_map = {
    ch : i for i, ch in enumerate(chs)
}
n_channels = len(chs)

# allocate metric results
n_participants = 9
task_distinct = np.zeros((n_participants, n_total_folds))
task_incon = np.zeros((n_participants, n_total_folds))
clsf_f1 = np.zeros((n_participants, n_total_folds))

# iterate over participants
for i_p, p in enumerate(range(1,n_participants+1)):
    print("*"*80)
    print(f"Participant {p}...")

    # Extract all data from both sessions
    X = np.concatenate([d[p][1]['trials'], d[p][2]['trials']])
    y = np.concatenate([d[p][1]['labels'], d[p][2]['labels']])
    blocks = np.concatenate([
        d[p][1]['run_labels'], 
        d[p][2]['run_labels']+(1+max(d[p][1]['run_labels'])) # shift the second session's block labels
    ])

    # split trials into task and rest
    length = 3 * Fs
    stride = 3 * Fs
    X = epoch(X, length, stride)
    Xtask = X[:,1,:,:]
    Xrest = X[:,0,:,:]
    y_rest = np.zeros((Xrest.shape[0],))

    # concatenate rest and task epochs
    X = np.concatenate([Xtask, Xrest])
    y = np.concatenate([y, y_rest])
    blocks = np.concatenate([blocks, blocks])

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
    n_trials, n_channels, n_samples = X.shape
    n_blks = len(np.unique(blocks))

    # block means
    block_means = np.zeros((n_blks, n_channels, n_channels))
    for i, block in enumerate(np.sort(np.unique(blocks))):
        block_covs = pyriemann.utils.covariance.covariances(X[blocks == block])
        block_means[i] = pyriemann.utils.mean.mean_covariance(block_covs)

    # Generate epochs for each trial
    length = 1
    stride = 0.5
    n_samples_epoch = int(Fs * length)
    X_epoched = epoch(X, n_samples_epoch, int(Fs * stride))
    n_epochs = X_epoched.shape[1]

    # apply the rebiasing to the epochs
    blocks_epochs = np.repeat(blocks, X_epoched.shape[1])

    X_epoched = np.reshape(X_epoched, (-1, n_channels, n_samples_epoch))
    X_epoch_covs = pyriemann.utils.covariance.covariances(X_epoched)
    X_epoch_covs = apply_rebias_to_groups(X_epoch_covs, blocks_epochs, block_means)

    # put the data back into trial, epoch, format
    X_epoch_covs = np.reshape(X_epoch_covs, (n_trials, n_epochs, n_channels, n_channels))

    # select the number of clusters
    # iterate over folds
    k_sel_criterion = np.zeros((2, len(krange), n_total_folds)) # first dimension is for task distinct and task inconsistency datasets (task and all)
    X_feats_fold = {
        'task' : {
            'tr' : [],
            'te' : []
        },
        'all' : {
            'tr' : [],
            'te' : []
        }
    }
    y_fold = {
        'task' : {
            'tr' : [],
            'te' : []
        },
        'all' : {
            'tr' : [],
            'te' : []
        }
    }

    print("Performing feature extraction and k selection...")
    for i_r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=i_r)
        for i_f, (tr_idx, te_idx) in enumerate(skf.split(X_epoch_covs, y)):
            print(f"\tParticipant {p}, repeat {i_r+1}, fold {i_f+1}")

            # split data into training and testing
            X_covs_tr = X_epoch_covs[tr_idx]
            y_tr = y[tr_idx]
            X_covs_te = X_epoch_covs[te_idx]
            y_te = y[te_idx]

            # extract the task trials
            task_trials = y_tr != 0
            X_task_covs_tr = X_covs_tr[task_trials]
            y_task_tr = y_tr[task_trials]
            X_task_covs_te = X_covs_te[y_te != 0]
            y_task_te = y_te[y_te != 0]

            # apply feature extraction
            X_covs_tr = np.reshape(X_covs_tr, (-1, n_channels, n_channels))
            X_task_covs_tr = np.reshape(
                X_task_covs_tr, (-1, n_channels, n_channels)
            )
            X_covs_te = np.reshape(X_covs_te, (-1, n_channels, n_channels))
            X_task_covs_te = np.reshape(
                X_task_covs_te, (-1, n_channels, n_channels)
            )

            feature_extractor = ScaledTangentSpace().fit(X_covs_tr)
            X_feats_tr = feature_extractor.transform(X_covs_tr)
            X_feats_te = feature_extractor.transform(X_covs_te)

            k_sel_criterion[1, :, i_r*n_folds + i_f] = cluster_pred_strength(
                X_feats_tr, 
                y=np.repeat(y_tr, n_epochs), 
                krange=krange,
                n_repeats=1
            )

            feature_extractor.fit(X_task_covs_tr)
            X_task_feats_tr = feature_extractor.transform(X_task_covs_tr)
            X_task_feats_te = feature_extractor.transform(X_task_covs_te)

            k_sel_criterion[0, :, i_r*n_folds + i_f] = cluster_pred_strength(
                X_task_feats_tr, 
                y=np.repeat(y_task_tr, n_epochs), 
                krange=krange, 
                n_repeats=1
            )


            # put the data back into trial, epoch, format
            X_feats_tr = np.reshape(X_feats_tr, (y_tr.shape[0], n_epochs, -1))
            X_feats_te = np.reshape(X_feats_te, (y_te.shape[0], n_epochs, -1))
            X_task_feats_tr = np.reshape(
                X_task_feats_tr, (y_task_tr.shape[0], n_epochs, -1)
            )
            X_task_feats_te = np.reshape(
                X_task_feats_te, (y_task_te.shape[0], n_epochs, -1)
            )

            # save the features for later
            X_feats_fold['task']['tr'].append(X_task_feats_tr)
            X_feats_fold['task']['te'].append(X_task_feats_te)
            X_feats_fold['all']['tr'].append(X_feats_tr)
            X_feats_fold['all']['te'].append(X_feats_te)
            y_fold['task']['tr'].append(y_task_tr)
            y_fold['task']['te'].append(y_task_te)
            y_fold['all']['tr'].append(y_tr)
            y_fold['all']['te'].append(y_te)

            # compute the classification performance
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            X_tr = X_tr[y_tr != 0]
            y_tr = y_tr[y_tr != 0]
            X_te = X_te[y_te != 0]
            y_te = y_te[y_te != 0]
            clsf = ms.classification.CSP_LDA(classes=4)
            clsf.fit(X_tr, y_tr)
            y_pred = clsf.predict(X_te)
            clsf_f1[i_p, i_r*n_folds + i_f] = f1_score(y_te, y_pred, average='macro')


    
    # Identify the number of clusters for both task and all data
    k_sel_criterion = np.mean(k_sel_criterion, axis=2) + np.std(k_sel_criterion, axis=2)/np.sqrt(n_total_folds)
    k_sel_thresh = 0.3
    K_all = min(krange)
    K_task = min(krange)
    for i, k in enumerate(krange):
        if k_sel_criterion[1, i] >= k_sel_thresh:
            K_all = k
        if k_sel_criterion[0, i] >= k_sel_thresh:
            K_task = k
    
    print(f"\tSelected number of clusters for all data: {K_all}")
    print(f"\tSelected number of clusters for task data: {K_task}")

    # iterate over folds to compute the metrics
    for i_r in range(n_repeats):
        for i_f in range(n_folds):

            # Perform clustering and generate the trajectory sub-space
            np.random.seed(42)
            all_state_space = ms.MarkovStateSpace(n_clusters=K_all).fit(
                X_feats_fold['all']['tr'][i_r*n_folds+i_f], verbose=True
            )

            # create a state space for the task data only
            task_state_space = ms.MarkovStateSpace(n_clusters=K_task).fit(
                X_feats_fold['task']['tr'][i_r*n_folds+i_f], verbose=True
            )

            # create Markov State Transition Matrix for each task
            y_task_te = y_fold['task']['te'][i_r*n_folds+i_f]
            labels = np.unique(y_task_te)
            models = [None] * len(labels)
            for i, label in enumerate(labels):
                Xlabel = X_feats_fold['task']['te'][i_r*n_folds+i_f][y_task_te == label]
                S = task_state_space.transform(Xlabel)
                models[i] = ms.MarkovChainModel(task_state_space).fit(
                    S, damping=0.015*K_task
                )

            # compute the taskDistinct
            task_distinct[i_p, i_r*n_folds + i_f] = ms.task_distinct(models)

            # create a Markov State Transition Matrix for all tasks and rest
            y_all_te = y_fold['all']['te'][i_r*n_folds+i_f]
            models = [None] * (len(labels) + 1)
            for i, label in enumerate(labels):
                Xlabel = X_feats_fold['all']['te'][i_r*n_folds+i_f][y_all_te == label]
                S = all_state_space.transform(Xlabel)
                models[i] = ms.MarkovChainModel(all_state_space).fit(
                    S, damping=0.015*K_all
                )
            S = all_state_space.transform(X_feats_fold['all']['te'][i_r*n_folds+i_f][y_all_te == 0])
            models[-1] = ms.MarkovChainModel(all_state_space).fit(
                S, damping=0.015*K_all
            )

            # Compute relativeTaskInconsistency
            task_incon[i_p, i_r*n_folds+i_f] = ms.relative_task_inconsistency(
                models[:-1], models[-1]
            )
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
for i, p in enumerate(range(1, n_participants+1)):
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