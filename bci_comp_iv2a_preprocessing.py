import numpy as np
from scipy import io
from glob import glob
import pickle
import os
import mne

## Define some parameters and paths
data_dir = 'data/BCI_Comp_IV_2a'
bandpass = [8, 30]
channels = np.arange(22)
save_dataset = True
dataset_file = f"{data_dir}/preprocessed_data.pkl"

# Identify the files to load
files = glob(f"{data_dir}/raw_data/A0*.gdf")

# Load and preprocess the data
dataset = {
    p : {
        s : {
            'trials' : None,
            'resting' : None,
            'labels' : None,
            'run_labels' : None
        } for s in (1,2)
    }
    for p in range(1, 10)
}

for file in files:
    _, filename = os.path.split(file)
    print(f"Preprocessing file {filename} ...")

    # extract the participant and session numbers
    p = int(filename[2])
    s = filename[3]
    
    if s == 'T':
        s_num = 1
    else:
        s_num = 2
    
    # Load the true labels
    label_file = f"{data_dir}/true_labels/A0{p}{s}.mat"
    labels = io.loadmat(label_file)['classlabel']
    labels = np.asarray([l[0] for l in labels])

    # Load the EEG data 
    data = mne.io.read_raw_gdf(file, preload=True, verbose=False)
    events, events_id = mne.events_from_annotations(data)
 
    # create run labels - 48 trials per run
    run_labels = np.concatenate([r*np.ones((48,)) for r in range(6)], axis=0)
        
    # bandpass filter the data
    data = data.filter(
        bandpass[0], bandpass[1],
        method='iir', iir_params={'order' : 4, 'ftype': 'butter'},
        verbose=False
    )
    
    if '276' in events_id:
        # extract the 2 min resting baseline data, if available
        resting = mne.Epochs(
            data, events, event_id=events_id['276'], 
            tmin=0, tmax=120, baseline=None, 
            preload=True, picks=channels,
            verbose=False
        ).get_data()

        resting = resting * (10**6) # convert to microvolts
    else:
        resting = None

    # extract the trials 
    trials = mne.Epochs(
        data, events, event_id=events_id['768'], 
        tmin=0, tmax=6, baseline=None, 
        preload=True, picks=channels,
        verbose=False
    ).get_data()

    trials = trials * (10**6) # convert to microvolts
    
    # find and remove rejected trials
    Nt = np.sum(events[:,2]==events_id['768'])

    clean = np.ones((Nt,),dtype=np.int16)
    trial_ts = events[events[:,2]==events_id['768'],0]
    rejects = events[events[:,2]==events_id['1023'],0]

    for reject_ts in rejects:
        clean -= (trial_ts==reject_ts).astype(np.int16)

    Nrejected = np.sum(clean==0)
    print(f"\tRejected {Nrejected} of {Nt} trials.")

    trials = trials[clean==1]
    labels = labels[clean==1]
    run_labels = run_labels[clean==1]
    
    # store the data
    dataset[p][s_num]['trials'] = trials
    dataset[p][s_num]['resting'] = resting
    dataset[p][s_num]['labels'] = labels
    dataset[p][s_num]['run_labels'] = run_labels


if save_dataset:
    with open(dataset_file, 'wb') as f:
        pickle.dump(dataset, f, protocol=pickle.HIGHEST_PROTOCOL)