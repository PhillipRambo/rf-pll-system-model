# src/rfmodel/rf/ofdm.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.random import get_rng
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class OFDMParams:
    fft_size: int                 # Nfft
    cp_len: int                   # cyclic prefix length [samples]
    n_used: int                   # number of active subcarriers
    n_symbols: int                # number of OFDM symbols in one frame
    bits_per_symbol: int = 2      # 2=QPSK, 4=16QAM
    fs_hz: float = 1.0            # sample rate of output waveform
    tx_power_w: float = 1.0       # target average output power


class OFDMTxBlock(Block):
    """
    OFDM transmitter block producing a complex baseband waveform.

    Output signal convention
    ------------------------
    - complex baseband / complex envelope
    - one frame containing n_symbols OFDM symbols
    - includes cyclic prefix

    Notes
    -----
    - First version supports QPSK and 16-QAM
    - Active subcarriers are centered around DC, but DC itself is left unused
    - No pilots, coding, sync, pulse shaping, or windowing yet
    """

    type_name = "ofdm_tx"

    def __init__(self, name: str, params: OFDMParams, seed: int | None = None):
        super().__init__(name=name)
        self.params = params
        self._rng = get_rng(seed)
        self._validate()

    def _validate(self) -> None:
        p = self.params

        if p.fft_size <= 0 or (p.fft_size & (p.fft_size - 1)) != 0:
            raise ValueError("fft_size must be a positive power of 2")

        if p.cp_len < 0:
            raise ValueError("cp_len must be >= 0")

        if p.n_used <= 0:
            raise ValueError("n_used must be > 0")

        # leave DC unused, so max active tones is Nfft - 1
        if p.n_used >= p.fft_size:
            raise ValueError("n_used must be < fft_size")

        if p.n_symbols <= 0:
            raise ValueError("n_symbols must be > 0")

        if p.bits_per_symbol not in (2, 4):
            raise ValueError("bits_per_symbol must be 2 (QPSK) or 4 (16-QAM)")

        if p.fs_hz <= 0:
            raise ValueError("fs_hz must be > 0")

        if p.tx_power_w <= 0:
            raise ValueError("tx_power_w must be > 0")

    def process(self, s: Signal | None = None) -> Signal:
        """
        Generate one OFDM frame and return it as a Signal.

        Parameters
        ----------
        s : ignored
            Present only to match Block-style APIs.
        """
        p = self.params

        n_bits = p.n_used * p.n_symbols * p.bits_per_symbol
        bits = self._rng.integers(0, 2, size=n_bits, endpoint=False)

        symbols = self._map_bits(bits, p.bits_per_symbol)
        symbols_2d = symbols.reshape(p.n_symbols, p.n_used)

        X = self._allocate_subcarriers(symbols_2d, p.fft_size, p.n_used)

        # numpy ifft includes 1/N scaling
        x_no_cp = np.fft.ifft(X, axis=1)

        if p.cp_len > 0:
            cp = x_no_cp[:, -p.cp_len:]
            x_blocks = np.concatenate([cp, x_no_cp], axis=1)
        else:
            x_blocks = x_no_cp

        x = x_blocks.reshape(-1)

        # normalize average waveform power to tx_power_w
        pwr = np.mean(np.abs(x) ** 2)
        if pwr > 0:
            x = x * np.sqrt(p.tx_power_w / pwr)

        # If your Signal class supports metadata, attach bits and params there.
        # Otherwise just return the waveform.
        return Signal(x=x, fs_hz=p.fs_hz)

    def _map_bits(self, bits: np.ndarray, bits_per_symbol: int) -> np.ndarray:
        if bits_per_symbol == 2:
            return self._map_qpsk(bits)
        if bits_per_symbol == 4:
            return self._map_16qam(bits)
        raise ValueError("Unsupported modulation")

    def _map_qpsk(self, bits: np.ndarray) -> np.ndarray:
        b = bits.reshape(-1, 2)

        # Gray-coded QPSK
        # 00 -> +1 + j
        # 01 -> -1 + j
        # 11 -> -1 - j
        # 10 -> +1 - j
        i = 1.0 - 2.0 * b[:, 1]
        q = 1.0 - 2.0 * b[:, 0]

        # Normalize to unit average symbol power
        return (i + 1j * q) / np.sqrt(2.0)

    def _map_16qam(self, bits: np.ndarray) -> np.ndarray:
        b = bits.reshape(-1, 4)

        # Gray-coded 16-QAM for I and Q separately
        # bit pairs: 00 -> +3, 01 -> +1, 11 -> -1, 10 -> -3
        def levels(msb: np.ndarray, lsb: np.ndarray) -> np.ndarray:
            out = np.empty_like(msb, dtype=float)

            mask_00 = (msb == 0) & (lsb == 0)
            mask_01 = (msb == 0) & (lsb == 1)
            mask_11 = (msb == 1) & (lsb == 1)
            mask_10 = (msb == 1) & (lsb == 0)

            out[mask_00] = 3.0
            out[mask_01] = 1.0
            out[mask_11] = -1.0
            out[mask_10] = -3.0
            return out

        i = levels(b[:, 0], b[:, 1])
        q = levels(b[:, 2], b[:, 3])

        # Normalize average symbol power to 1
        return (i + 1j * q) / np.sqrt(10.0)

    def _allocate_subcarriers(
        self,
        data_symbols: np.ndarray,
        fft_size: int,
        n_used: int,
    ) -> np.ndarray:
        """
        Map active subcarriers around DC, leaving DC unused.

        Output shape: (n_symbols, fft_size)

        Layout in FFT bins:
        - positive frequencies go in bins [1 : n_pos+1]
        - negative frequencies go in bins [-n_neg : ]
        - DC bin 0 left as zero
        """
        n_ofdm_symbols = data_symbols.shape[0]
        X = np.zeros((n_ofdm_symbols, fft_size), dtype=complex)

        n_pos = n_used // 2
        n_neg = n_used - n_pos

        # split data: first half -> negative freq bins, second half -> positive freq bins
        neg_data = data_symbols[:, :n_neg]
        pos_data = data_symbols[:, n_neg:]

        if n_pos > 0:
            X[:, 1 : 1 + n_pos] = pos_data

        if n_neg > 0:
            X[:, -n_neg:] = neg_data

        return X