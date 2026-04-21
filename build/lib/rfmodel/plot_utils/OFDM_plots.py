import matplotlib.pyplot as plt
import numpy as np
from contextlib import contextmanager


@contextmanager
def _plot_style(science: bool, fig_scale: float = 1.0):
    """Apply IEEE scienceplots style when science=True, else default matplotlib."""
    if science:
        try:
            import scienceplots  # noqa: F401
        except ImportError:
            raise ImportError(
                "scienceplots is not installed. Run: pip install scienceplots"
            )
        with plt.style.context(["science", "ieee", "no-latex"]):
            base = plt.rcParams["figure.figsize"]
            plt.rcParams.update({
                "figure.dpi": 300,
                "figure.figsize": [s * fig_scale for s in base],
            })
            yield
    else:
        yield


def _figsize(default_w, default_h, science, fig_scale):
    if science:
        return {}
    return {"figsize": (default_w, default_h)}


def _save_and_show(save_path):
    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    plt.show()


def plot_bits(bits, n_bits=80, title="Input bits", science=False, fig_scale=1,
              save_path=None):
    bits = np.asarray(bits).astype(int)
    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(10, 2.5, science, fig_scale))
        ax.step(np.arange(min(n_bits, len(bits))), bits[:n_bits], where="mid",
                color="black" if science else None)
        ax.set_ylim(-0.2, 1.2)
        ax.set_xlabel("Bit index")
        ax.set_ylabel("Bit")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _save_and_show(save_path)


def plot_constellation(x, title="QAM constellation", science=False, fig_scale=1,
                       save_path=None):
    x = np.asarray(x)
    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(5, 5, science, fig_scale))
        ax.plot(np.real(x), np.imag(x), ".",
                color="black" if science else None)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal")
        plt.tight_layout()
        _save_and_show(save_path)


def plot_ofdm_frequency_bins(Xk, title="OFDM frequency bins (raw FFT order)",
                              science=False, fig_scale=1, save_path=None):
    Xk = np.asarray(Xk)
    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(10, 4, science, fig_scale))
        markerline, stemlines, baseline = ax.stem(
            np.arange(len(Xk)), np.abs(Xk), basefmt=" "
        )
        if science:
            plt.setp(stemlines, color="black")
            plt.setp(markerline, color="black")
        ax.set_xlabel("FFT bin index")
        ax.set_ylabel("|X[k]|")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _save_and_show(save_path)


def plot_ofdm_frequency_bins_centered(Xk, title="OFDM frequency bins (centered)",
                                       science=False, fig_scale=1, save_path=None):
    Xk = np.asarray(Xk)
    Xc = np.fft.fftshift(Xk)
    N = len(Xk)
    k = np.arange(-N // 2, N // 2) if N % 2 == 0 else np.arange(-(N // 2), N // 2 + 1)

    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(10, 4, science, fig_scale))
        markerline, stemlines, baseline = ax.stem(k, np.abs(Xc), basefmt=" ")
        if science:
            plt.setp(stemlines, color="black")
            plt.setp(markerline, color="black")
        ax.set_xlabel("Subcarrier index")
        ax.set_ylabel("|X[k]|")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _save_and_show(save_path)


def plot_time_signal(x, n_samples=200, title="Time-domain signal",
                     science=False, fig_scale=1, save_path=None):
    x = np.asarray(x)
    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(10, 4, science, fig_scale))
        if science:
            ax.plot(np.real(x[:n_samples]), label="Real", color="black")
            ax.plot(np.imag(x[:n_samples]), label="Imag", color="black", linestyle="--")
        else:
            ax.plot(np.real(x[:n_samples]), label="Real")
            ax.plot(np.imag(x[:n_samples]), label="Imag", alpha=0.8)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Amplitude")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        plt.tight_layout()
        _save_and_show(save_path)


def plot_spectrum(x, fs_hz=1.0, title="Spectrum", science=False, fig_scale=1,
                  save_path=None):
    x = np.asarray(x)
    N = len(x)
    X = np.fft.fftshift(np.fft.fft(x))
    f = np.fft.fftshift(np.fft.fftfreq(N, d=1 / fs_hz))
    mag_db = 20 * np.log10(np.abs(X) + 1e-12)

    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(10, 4, science, fig_scale))
        ax.plot(f / 1e6, mag_db, color="black" if science else None)
        ax.set_xlabel("Frequency [MHz]")
        ax.set_ylabel("Magnitude [dB]")
        ax.set_title(title)
        ax.grid(True)
        plt.tight_layout()
        _save_and_show(save_path)


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
    science=False,
    fig_scale=1,
    save_path=None,
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
    science : bool
        If True, apply IEEE scienceplots style (black & white, publication ready).
    fig_scale : float
        Scale factor applied to figure size. Only used when science=True.
    save_path : str | None
        If provided, save the figure to this path before showing.
    """
    symbols = np.asarray(symbols)

    with _plot_style(science, fig_scale):
        fig, ax = plt.subplots(**_figsize(7, 7, science, fig_scale))
        ax.plot(np.real(symbols), np.imag(symbols), ".",
                alpha=0.5, color="black" if science else None)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.set_title(title)
        ax.grid(True)
        ax.set_aspect("equal")

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
                    ax.annotate(
                        label,
                        (xr, xi),
                        textcoords="offset points",
                        xytext=(0, 5),
                        fontsize=6,
                        ha="center",
                    )
            else:
                for s, label in zip(symbols, labels):
                    ax.annotate(
                        label,
                        (np.real(s), np.imag(s)),
                        textcoords="offset points",
                        xytext=(0, 5),
                        fontsize=6,
                        ha="center",
                    )

        plt.tight_layout()
        _save_and_show(save_path)
