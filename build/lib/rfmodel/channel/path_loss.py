# src/rfmodel/channel/pathloss.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class PathLossParams:
    freq_hz: float
    distance_m: float
    tx_gain_db: float = 0.0 # Antenna Gain
    rx_gain_db: float = 0.0 # Antenna Gain


class PathLossBlock(Block):
    """
    Free-space path loss using Friis equation.
    """

    type_name = "pathloss"

    def __init__(self, name: str, params: PathLossParams):
        super().__init__(name=name)
        self.params = params

    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x

        c = 299792458.0
        wavelength = c / p.freq_hz

        # Linear antenna gains
        Gt = 10 ** (p.tx_gain_db / 10.0)
        Gr = 10 ** (p.rx_gain_db / 10.0)

        # Friis power gain
        G = Gt * Gr * (wavelength / (4 * np.pi * p.distance_m)) ** 2

        # Convert to amplitude scaling
        alpha = np.sqrt(G)

        y = alpha * x
        return s.copy_with(x=y)