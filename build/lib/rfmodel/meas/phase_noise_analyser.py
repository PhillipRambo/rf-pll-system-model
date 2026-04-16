import scipy.signal as signal
import numpy as np

def calculate_phase_noise_curve(
    y,
    fs,
    nperseg=2**14,
    one_sided=False,
    detrend_type='linear',
    window='hann',
    target_offset=1e6,
    band=10,
):
    """
    Calculates Phase noise, based by unwrapping the phase of the given signal and performing
    Welch method to get the PSD.

    Args:
        y: Input signal
        fs: Sampling frequency
        nperseg: Number of samples per Welch segment (sets frequency resolution)
        one_sided: If True, return one-sided spectrum only
        detrend_type: Detrending type passed to scipy.signal.detrend ('linear', 'constant', False)
        window: Window function for Welch method (e.g. 'hann', 'blackman')
        target_offset: Frequency offset at which to report phase noise in Hz (default 1 MHz)
        band: Number of bins on each side of target_offset to average over

    Returns:
        (offsets_pe, L_f, pn_at_offset):
            offsets_pe: Frequency offset array [Hz]
            L_f: Phase noise array [dBc/Hz]
            pn_at_offset: Phase noise at target_offset averaged over ±band bins [dBc/Hz]
    """
    phi = np.unwrap(np.angle(y))
    phi_detrended = signal.detrend(phi, type=detrend_type)

    f_pn, S_phi = signal.welch(phi_detrended, fs, window=window, nperseg=nperseg, scaling='density')

    pos = f_pn > 0
    offsets_pe = f_pn[pos]
    L_f = 10 * np.log10(S_phi[pos])

    idx = np.argmin(np.abs(offsets_pe - target_offset))
    pn_at_offset = np.mean(L_f[idx - band : idx + band + 1])
    print(f"Phase noise at {target_offset/1e6:.3f} MHz offset: {pn_at_offset:.2f} dBc/Hz")

    return offsets_pe, L_f, pn_at_offset
