import scipy.signal as signal
import numpy as np

def calculate_phase_noise_curve(y, fs, nperseg=2**18, one_sided=False):
    """
    Calculates the SSB Phase Noise curve L(f).
    
    Args:
        y: Time domain signal (complex or real)
        fs: Sampling frequency
        nperseg: Length of each segment for Welch's method
        one_sided: If True, returns positive frequencies only. 
                   If False (default), handles complex/centered spectra.
    """
    # 1. Estimate PSD
    f, psd = signal.welch(y, fs, window='hann', nperseg=nperseg, 
                          scaling='density', return_onesided=one_sided)
    
    # 2. Handle Frequency Alignment
    if not one_sided:
        # For complex signals, we center the carrier at 0 Hz index
        f = np.fft.fftshift(f)
        psd = np.fft.fftshift(psd)
    
    # 3. Find the carrier
    carrier_idx = np.argmax(psd)
    f_carrier = f[carrier_idx]
    
    # 4. Integrate Carrier Power
    df = f[1] - f[0] 
    # Use a small integration window around the peak
    p_carrier_total = np.sum(psd[carrier_idx-5:carrier_idx+6]) * df
    
    # 5. Calculate Offsets and Phase Noise
    # We grab everything to the right of the carrier peak
    offsets = f[carrier_idx+1:] - f_carrier
    psd_sideband = psd[carrier_idx+1:]
    
    # L(f) calculation
    pn_curve = 10 * np.log10(psd_sideband / p_carrier_total)
    
    return offsets, pn_curve