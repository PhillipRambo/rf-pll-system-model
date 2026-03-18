from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class OFDMParams:
    n_fft: int
    cp_len: int
    n_data_subcarriers: int | None = None
    normalize_ifft: bool = False


class OFDMModulator(Block):
    """
    OFDM modulator.

    Input
    -----
    s.x : 1D complex array of QAM symbols

    Output
    ------
    Time-domain OFDM waveform in s.x

    Parameters
    ----------
    n_fft : int
        IFFT size / total number of subcarriers.
    cp_len : int
        Cyclic prefix length in samples.
    n_data_subcarriers : int | None
        Number of active data subcarriers.
        If None, all n_fft subcarriers are used.
    normalize_ifft : bool
        If True, multiply IFFT output by sqrt(n_fft) so average
        time-domain power roughly matches average active-subcarrier power.
    """

    type_name = "ofdm_modulator"

    def __init__(self, name: str, params: OFDMParams):
        super().__init__(name=name)
        self.params = params

        if params.n_fft <= 0:
            raise ValueError("n_fft must be > 0")
        if params.cp_len < 0:
            raise ValueError("cp_len must be >= 0")

        if params.n_data_subcarriers is None:
            self.n_data = params.n_fft
        else:
            self.n_data = params.n_data_subcarriers

        if self.n_data <= 0 or self.n_data > params.n_fft:
            raise ValueError("n_data_subcarriers must satisfy 1 <= n_data <= n_fft")

    def process(self, s: Signal) -> Signal:
        x = np.asarray(s.x)

        if x.ndim != 1:
            raise ValueError("OFDMModulator expects s.x to be a 1D complex symbol array")

        if not np.iscomplexobj(x):
            raise ValueError("OFDMModulator expects complex QAM symbols as input")

        n_fft = self.params.n_fft
        cp_len = self.params.cp_len
        n_data = self.n_data

        if len(x) % n_data != 0:
            raise ValueError(
                f"Number of input QAM symbols ({len(x)}) must be a multiple of "
                f"n_data_subcarriers ({n_data})"
            )

        # Shape: (n_ofdm_symbols, n_data_subcarriers)
        qam_blocks = x.reshape(-1, n_data)

        time_blocks = []

        for qam_block in qam_blocks:
            # Frequency-domain OFDM bin vector
            Xk = np.zeros(n_fft, dtype=np.complex128)

            # put data in the first n_data bins directly
            Xk[:n_data] = qam_block

            # Time-domain OFDM symbol
            xn = np.fft.ifft(Xk)

            if self.params.normalize_ifft:
                xn = xn * np.sqrt(n_fft)

            # Add cyclic prefix
            if cp_len > 0:
                cp = xn[-cp_len:]
                xn = np.concatenate([cp, xn])

            time_blocks.append(xn)

        y = np.concatenate(time_blocks)

        meta = dict(s.meta) if s.meta is not None else {}
        meta.update(
            {
                "modulation": "OFDM",
                "n_fft": n_fft,
                "cp_len": cp_len,
                "n_data_subcarriers": n_data,
                "normalize_ifft": self.params.normalize_ifft,
            }
        )

        return s.copy_with(x=y, meta=meta)
    

"""
For future make guard frequencies and DC nulls
"""
    
import numpy as np
from rfmodel.core.signal import Signal
from rfmodel.QAM_OFDM.QAM_modulator import QAMModulator, QAMParams
rng = np.random.default_rng(0)

# 64-QAM: 6 bits/symbol
n_qam_symbols = 128
bits = rng.integers(0, 2, size=n_qam_symbols * 6, dtype=np.uint8)

sig_bits = Signal(x=bits, fs_hz=20e6, fc_hz=0.0, meta={})

qam = QAMModulator("qam1", QAMParams(M=64, gray_map=True, unit_average_power=True))
sig_qam = qam.process(sig_bits)

ofdm = OFDMModulator(
    "ofdm1",
    OFDMParams(
        n_fft=64,
        cp_len=16,
        n_data_subcarriers=64,
        normalize_ifft=True,
    ),
)

sig_tx = ofdm.process(sig_qam)

print("QAM symbols:", len(sig_qam.x))
print("OFDM waveform samples:", len(sig_tx.x))
