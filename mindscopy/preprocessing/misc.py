import numpy as np
from scipy.signal import butter, filtfilt, resample

def epoch(X, length, stride):
    """
    Segment the signal X into (sub)-epochs of the given length and stride.

    Parameters
    ----------
    X : array_like (n_trials, n_channels, n_samples)
        The input signal to be segmented, where n_trials is the number of epochs to
        segment into smaller epochs, n_channels is 
        the number of channels and n_samples is the number of samples. If data has
        not been segmented into epochs, n_trials should be 1.

    length : int
        The length in samples of the epochs to segment the signal into.
        Should be equal to the sampling rate of the signal multiplied by the
        desired epoch length in seconds.

    stride : int
        The stride in samples to use when segmenting the signal into epochs.
        Should be equal to the sampling rate of the signal multiplied by the
        desired stride in seconds.

    Returns
    -------
    X_epoch : array_like (n_trials, n_epochs, n_channels, length)
        The segmented signal, where n_trials is the number of trials, n_epochs is the number
        of epochs per trial, n_channels is the number of channels and length is the
        length of each epoch.
    """
    n_trials, n_channels, n_samples = X.shape

    n_epochs = (n_samples - length) // stride + 1

    X_epoch = np.zeros((n_trials, n_epochs, n_channels, length))

    for i in range(n_epochs):
        X_epoch[:, i] = X[:, :, i*stride:i*stride+length]

    return X_epoch


def bandpass_filter(X, lowcut, highcut, fs, order=4):
    """
    Bandpass filter the signal X using a Butterworth filter with the given
    cutoff frequencies and sampling rate. Using the sos filter implementation.

    Parameters
    ----------
    X : array_like (n_channels, n_samples)
        The input signal to be filtered, where n_channels is the number of channels
        and n_samples is the number of samples.

    lowcut : float
        The low cutoff frequency in Hz.
    
    highcut : float
        The high cutoff frequency in Hz.

    fs : float
        The sampling rate of the signal in Hz.

    order : int, optional
        The order of the Butterworth filter. Default is 4.

    Returns
    -------
    X_filt : array_like (n_channels, n_samples)
        The filtered signal.
    """

    sos = butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    X_filt = filtfilt(sos, X)

    return X_filt

def downsample(X, fs_new, fs_old):
    """
    Downsample the signal X from the old sampling rate to the new sampling rate.

    Parameters
    ----------
    X : array_like (n_channels, n_samples)
        The input signal to be resampled, where n_channels is the number of channels
        and n_samples is the number of samples.

    fs_new : float
        The new sampling rate in Hz.

    fs_old : float
        The old sampling rate in Hz.

    Returns
    -------
    X_resampled : array_like (n_channels, n_samples_new)
        The resampled signal.
    """
    n_channels, n_samples = X.shape

    n_samples_new = int(n_samples * fs_new / fs_old)

    X_resampled = resample(X, n_samples_new, axis=1)

    return X_resampled