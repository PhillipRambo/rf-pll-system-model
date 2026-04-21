# src/rfmodel/channel/awgn.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.random import get_rng
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class AWGNParams:
    snr_db: float


class AWGNBlock(Block):
    """
    Complex-baseband AWGN channel.

    Adds proper complex Gaussian noise to the input signal such that

        SNR = Ps / Pn

    where:
    - Ps = mean signal power = E[|x|^2]
    - Pn = mean added noise power = E[|n|^2]

    Notes
    -----
    - Noise is set from the instantaneous average power of the input block.
    - This is a simple waveform-level SNR model, not Eb/N0.
    - No path loss, fading, delay, or Doppler.
    """

    type_name = "awgn"

    def __init__(self, name: str, params: AWGNParams, seed: int | None = None):
        super().__init__(name=name)
        self.params = params
        self._rng = get_rng(seed)

    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x

        # Average input signal power
        Ps = np.mean(np.abs(x) ** 2)

        # Convert SNR from dB to linear
        snr_linear = 10.0 ** (p.snr_db / 10.0)

        # Required complex noise power
        Pn = Ps / snr_linear

        # Proper complex Gaussian:
        # E[|n|^2] = 2*sigma^2  => sigma = sqrt(Pn/2)
        sigma = np.sqrt(Pn / 2.0)

        n = (
            self._rng.normal(0.0, sigma, size=x.shape)
            + 1j * self._rng.normal(0.0, sigma, size=x.shape)
        )

        y = x + n
        return s.copy_with(x=y)
    
