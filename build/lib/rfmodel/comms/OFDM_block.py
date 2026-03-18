from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class OFDMParams:
    n_fft: int
    cp_len: int
    n_data_subcarriers: int
    normalize_ifft: bool = False
    null_dc: bool = True


class OFDMModulator(Block):
    type_name = "ofdm_modulator"

    def __init__(self, name: str, params: OFDMParams):
        super().__init__(name=name)
        self.params = params

        if params.n_fft <= 0:
            raise ValueError("n_fft must be > 0")
        if params.cp_len < 0:
            raise ValueError("cp_len must be >= 0")
        if params.n_data_subcarriers <= 0:
            raise ValueError("n_data_subcarriers must be > 0")

        self.n_fft = params.n_fft
        self.cp_len = params.cp_len
        self.n_data = params.n_data_subcarriers

        self.active_bins = self._make_active_bins()

    def _make_active_bins(self) -> np.ndarray:
        n_fft = self.n_fft
        n_data = self.n_data

        if self.params.null_dc:
            if n_data % 2 != 0:
                raise ValueError("n_data_subcarriers must be even when null_dc=True")
            if n_data > n_fft - 1:
                raise ValueError("Too many data subcarriers for DC-null allocation")

            half = n_data // 2

            pos_bins = np.arange(1, half + 1)      # 1,2,...,half
            neg_bins = np.arange(n_fft - half, n_fft)  # n_fft-half,...,n_fft-1

            # Order matches input qam_block
            return np.concatenate([neg_bins, pos_bins])

        if n_data > n_fft:
            raise ValueError("n_data_subcarriers cannot exceed n_fft")

        return np.arange(n_data)

    def process(self, s: Signal) -> Signal:
        x = np.asarray(s.x)

        if x.ndim != 1:
            raise ValueError("OFDMModulator expects s.x to be a 1D complex symbol array")
        if not np.iscomplexobj(x):
            raise ValueError("OFDMModulator expects complex QAM symbols as input")
        if len(x) % self.n_data != 0:
            raise ValueError(
                f"Number of input QAM symbols ({len(x)}) must be a multiple of "
                f"n_data_subcarriers ({self.n_data})"
            )

        qam_blocks = x.reshape(-1, self.n_data)
        time_blocks = []

        for qam_block in qam_blocks:
            Xk = np.zeros(self.n_fft, dtype=np.complex128)
            Xk[self.active_bins] = qam_block

            xn = np.fft.ifft(Xk)

            if self.params.normalize_ifft:
                xn = xn * np.sqrt(self.n_fft)

            if self.cp_len > 0:
                xn = np.concatenate([xn[-self.cp_len:], xn])

            time_blocks.append(xn)

        y = np.concatenate(time_blocks)

        meta = dict(s.meta) if s.meta is not None else {}
        meta.update({
            "modulation": "OFDM",
            "n_fft": self.n_fft,
            "cp_len": self.cp_len,
            "n_data_subcarriers": self.n_data,
            "normalize_ifft": self.params.normalize_ifft,
            "active_bins": self.active_bins.tolist(),
        })

        return s.copy_with(x=y, meta=meta)