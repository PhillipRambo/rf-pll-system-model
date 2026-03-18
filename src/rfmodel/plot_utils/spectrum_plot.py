import numpy as np
import matplotlib.pyplot as plt 

def plot_top_spectrum(freq,
                      P_bin_W,
                      n_peaks=4,
                      margin_ratio=0.05,
                      full_view=False):
    """
    Plot power spectrum in dBm.

    full_view=False  → zoom around strongest components and show legend.
    full_view=True   → show full span, no legend.
    """

    # Convert to dBm
    P_dBm = 10*np.log10(np.maximum(P_bin_W, 1e-30)/1e-3)

    # Sort by frequency
    idx = np.argsort(freq)
    freq = freq[idx]
    P_dBm = P_dBm[idx]

    # --- Find strongest bins (suppress neighbors) ---
    P_copy = P_dBm.copy()
    peak_indices = []

    for _ in range(n_peaks):
        k = np.argmax(P_copy)
        peak_indices.append(k)

        kmin = max(0, k-2)
        kmax = min(len(P_copy), k+3)
        P_copy[kmin:kmax] = -np.inf

    peak_freqs = freq[peak_indices]
    peak_powers = P_dBm[peak_indices]

    # --- Plot ---
    plt.figure(figsize=(9, 4.5))
    plt.plot(freq/1e6, P_dBm, linewidth=1.2)

    plt.xlabel("Frequency (MHz)")
    plt.ylabel("Power (dBm)")
    plt.grid(True)

    if full_view:
        # Full span, no legend, no peak markers
        plt.xlim(freq.min()/1e6, freq.max()/1e6)
    else:
        # Highlight peaks
        for f, p in zip(peak_freqs, peak_powers):
            plt.plot(f/1e6, p, 'o',
                     label=f"{f/1e6:.6f} MHz, {p:.2f} dBm")

        f_min = np.min(peak_freqs)
        f_max = np.max(peak_freqs)
        span = f_max - f_min
        if span == 0:
            span = abs(f_min) if f_min != 0 else 1.0
        margin = span * margin_ratio

        plt.xlim((f_min - margin)/1e6, (f_max + margin)/1e6)
        plt.legend(fontsize=8)

    ymax = np.max(P_dBm)
    plt.ylim(ymax - 100, ymax + 5)

    plt.tight_layout()
    plt.show()
