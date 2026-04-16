import matplotlib.pyplot as plt
import numpy as np

def plot_bits(bits, n_bits=80, title="Input bits"):
    bits = np.asarray(bits).astype(int)
    plt.figure(figsize=(10, 2.5))
    plt.step(np.arange(min(n_bits, len(bits))), bits[:n_bits], where="mid")
    plt.ylim(-0.2, 1.2)
    plt.xlabel("Bit index")
    plt.ylabel("Bit")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_constellation(x, ref=None, title="QAM constellation", show_evm=False):
    x = np.asarray(x)

    plt.figure(figsize=(5, 5))
    plt.plot(np.real(x), np.imag(x), ".", alpha=0.6)

    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")

    if show_evm:
        if ref is None:
            raise ValueError("EVM requires a reference signal (ref).")

        ref = np.asarray(ref)

        # Align lengths (basic safety; better alignment may be needed in practice)
        n = min(len(x), len(ref))
        x = x[:n]
        ref = ref[:n]

        error = x - ref

        rms_ref = np.sqrt(np.mean(np.abs(ref)**2))
        rms_err = np.sqrt(np.mean(np.abs(error)**2))

        evm = rms_err / rms_ref
        evm_db = 20 * np.log10(evm + 1e-12)

        print(f"EVM (RMS): {evm:.6f}")
        print(f"EVM (dB):  {evm_db:.2f} dB")

    plt.show()


def plot_ofdm_frequency_bins(Xk, title="OFDM frequency bins (raw FFT order)"):
    Xk = np.asarray(Xk)

    plt.figure(figsize=(10, 4))
    plt.stem(np.arange(len(Xk)), np.abs(Xk), basefmt=" ")
    plt.xlabel("FFT bin index")
    plt.ylabel("|X[k]|")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_ofdm_frequency_bins_centered(Xk, title="OFDM frequency bins (centered)"):
    Xk = np.asarray(Xk)
    Xc = np.fft.fftshift(Xk)
    N = len(Xk)

    if N % 2 == 0:
        k = np.arange(-N // 2, N // 2)
    else:
        k = np.arange(-(N // 2), N // 2 + 1)

    plt.figure(figsize=(10, 4))
    plt.stem(k, np.abs(Xc), basefmt=" ")
    plt.xlabel("Subcarrier index")
    plt.ylabel("|X[k]|")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_time_signal(x, n_samples=200, title="Time-domain signal"):
    x = np.asarray(x)

    plt.figure(figsize=(10, 4))
    plt.plot(np.real(x[:n_samples]), label="Real")
    plt.plot(np.imag(x[:n_samples]), label="Imag", alpha=0.8)
    plt.xlabel("Sample index")
    plt.ylabel("Amplitude")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_spectrum(x, fs_hz=1.0, title="Spectrum"):
    x = np.asarray(x)
    N = len(x)

    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs_hz))
    mag_db = 20 * np.log10(np.abs(X) + 1e-12)

    plt.figure(figsize=(10, 4))
    plt.plot(f / 1e6, mag_db)
    plt.xlabel("Frequency [MHz]")
    plt.ylabel("Magnitude [dB]")
    plt.title(title)
    plt.grid(True)
    plt.show()


def reconstruct_one_ofdm_symbol_freq(qam_block, active_bins, n_fft):
    Xk = np.zeros(n_fft, dtype=np.complex128)
    Xk[np.asarray(active_bins)] = qam_block
    return Xk

def plot_constellation_with_bits(
    symbols,
    bits=None,
    bits_per_symbol=None,
    title="QAM constellation",
    annotate_unique=True,
):
    """
    Plot constellation points and optionally label ideal/unique points
    with their corresponding bit words.

    Parameters
    ----------
    symbols : array-like
        Complex symbols to plot.
    bits : array-like | None
        Flat bit array corresponding to 'symbols'. Length must be
        len(symbols) * bits_per_symbol if provided.
    bits_per_symbol : int | None
        Number of bits per symbol, e.g. 6 for 64-QAM.
    title : str
        Plot title.
    annotate_unique : bool
        If True, annotate only one label per unique constellation point.
    """
    symbols = np.asarray(symbols)

    plt.figure(figsize=(7, 7))
    plt.plot(np.real(symbols), np.imag(symbols), ".", alpha=0.5)
    plt.xlabel("In-phase")
    plt.ylabel("Quadrature")
    plt.title(title)
    plt.grid(True)
    plt.axis("equal")

    if bits is not None:
        bits = np.asarray(bits).astype(int)

        if bits_per_symbol is None:
            raise ValueError("bits_per_symbol must be provided when bits is provided")

        if len(bits) != len(symbols) * bits_per_symbol:
            raise ValueError(
                f"Expected {len(symbols) * bits_per_symbol} bits, got {len(bits)}"
            )

        bit_words = bits.reshape(len(symbols), bits_per_symbol)
        labels = ["".join(map(str, bw)) for bw in bit_words]

        if annotate_unique:
            seen = {}
            for s, label in zip(symbols, labels):
                key = (np.round(np.real(s), 12), np.round(np.imag(s), 12))
                if key not in seen:
                    seen[key] = label

            for (xr, xi), label in seen.items():
                plt.annotate(
                    label,
                    (xr, xi),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )
        else:
            for s, label in zip(symbols, labels):
                plt.annotate(
                    label,
                    (np.real(s), np.imag(s)),
                    textcoords="offset points",
                    xytext=(5, 5),
                    fontsize=8,
                )

    plt.show()
