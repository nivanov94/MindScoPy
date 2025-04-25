import numpy as np
from scipy.signal import butter, filtfilt, resample

def epoch(X, length, stride):
    """
    Segment the signal X into (sub)-epochs of the given length and stride.

    Parameters
    ----------
    X : array_like (Nt, Nc, Ns)
        The input signal to be segmented, where Nt is the number of epochs to
        segment into smaller epochs, Nc is 
        the number of channels and Ns is the number of samples. If data has
        not been segmented into epochs, Nt should be 1.

    length : int
        The length in samples of the sub-epochs to segment the signal into.
        Should be equal to the sampling rate of the signal multiplied by the
        desired epoch length in seconds.

    stride : int
        The stride in samples to use when segmenting the signal into sub-epochs.
        Should be equal to the sampling rate of the signal multiplied by the
        desired stride in seconds.

    Returns
    -------
    X_epoch : array_like (Nt, Ne, Nc, length)
        The segmented signal, where Nt is the number of epochs, Ne is the number
        of sub-epochs per epoch, Nc is the number of channels and length is the
        length of each sub-epoch.
    """
    Nt, Nc, Ns = X.shape

    Ne = (Ns - length) // stride + 1

    X_epoch = np.zeros((Nt, Ne, Nc, length))

    for i in range(Ne):
        X_epoch[:, i] = X[:, :, i*stride:i*stride+length]

    return X_epoch


def bandpass_filter(X, lowcut, highcut, fs, order=4):
    """
    Bandpass filter the signal X using a Butterworth filter with the given
    cutoff frequencies and sampling rate. Using the sos filter implementation.

    Parameters
    ----------
    X : array_like (Nc, Ns)
        The input signal to be filtered, where Nc is the number of channels
        and Ns is the number of samples.

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
    X_filt : array_like (Nc, Ns)
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
    X : array_like (Nc, Ns)
        The input signal to be resampled, where Nc is the number of channels
        and Ns is the number of samples.

    fs_new : float
        The new sampling rate in Hz.

    fs_old : float
        The old sampling rate in Hz.

    Returns
    -------
    X_resampled : array_like (Nc, Ns_new)
        The resampled signal.
    """
    Nc, Ns = X.shape

    Ns_new = int(Ns * fs_new / fs_old)

    X_resampled = resample(X, Ns_new, axis=1)

    return X_resampled