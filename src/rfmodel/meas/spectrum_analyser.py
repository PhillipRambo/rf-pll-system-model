import numpy as np

def spectrum_analyser(x, fs, tol=1e-12):
    """
    Compute one-sided power spectrum (W per FFT bin) of a real signal.

    - Uses rFFT.
    - Returns power per bin (not PSD).
    - Applies one-sided correction (doubles non-DC/Nyquist bins).
    - Prints a warning if dominant tone appears non-coherently sampled
      (energy present in adjacent bins).

    Parameters
    ----------
    x : ndarray
        Real-valued time samples.
    fs : float
        Sampling frequency [Hz].
    Rref : float
        Reference resistance [Ohm].
    tol : float
        Neighbor/peak power ratio threshold for leakage warning.

    Returns
    -------
    P_bin_W : ndarray
        Power per FFT bin [W].
    freq : ndarray
        Corresponding frequency vector [Hz].
    """
    N = len(x)

    if np.iscomplexobj(x):
        # Complex signal → full FFT, no symmetry
        X = np.fft.fft(x)
        freq = np.fft.fftfreq(N, 1/fs)
        P_bin_W = (np.abs(X)**2) / (N**2)

    else:
        # Real signal → one-sided FFT
        X = np.fft.rfft(x)
        freq = np.fft.rfftfreq(N, 1/fs)

        P_bin_W = (np.abs(X)**2) / (N**2)

        # One-sided correction
        if N % 2 == 0:
            P_bin_W[1:-1] *= 2.0
        else:
            P_bin_W[1:] *= 2.0

        # Coherent sampling check (dominant tone only)
        k_peak = np.argmax(np.abs(X))
        if 0 < k_peak < len(X)-1:
            neighbor_power = np.abs(X[k_peak-1])**2 + np.abs(X[k_peak+1])**2
            peak_power = np.abs(X[k_peak])**2
            if neighbor_power / peak_power > tol:
                print("Signal is not coherently sampled — expect spectral leakage.")

    return P_bin_W, freq