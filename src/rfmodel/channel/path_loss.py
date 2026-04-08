# src/rfmodel/channel/pathloss.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block
import warnings


@dataclass
class PathLossParams:
    freq_hz: float
    distance_m: float
    tx_ant_gain_db: float = 0.0 # Antenna Gain
    rx_ant_gain_db: float = 0.0 # Antenna Gain


class PathLossBlock(Block):
    """
    Free-space path loss using Friis equation with physical gain clamping
    and Far-Field validity checks.
    """

    type_name = "pathloss"

    def __init__(self, name: str, params: PathLossParams):
        super().__init__(name=name)
        self.params = params
        
        # Pre-calculate wavelength for the warning
        c = 299792458.0
        self.wavelength = c / params.freq_hz
        
        # --- Far-Field Warning Logic ---
        # A common rule of thumb is that Far-Field starts at d > 2*lambda
        # (or 2*D^2/lambda for large antennas)
        if params.distance_m < 2 * self.wavelength:
            warnings.warn(
                f"[{self.name}] Distance ({params.distance_m:.2e}m) is in the Near-Field "
                f"(< 2 * lambda = {2*self.wavelength:.2f}m). "
                "Friis gain has been clamped to Gt*Gr to simulate a lossless connection.",
                UserWarning
            )

    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x

        # Linear antenna gains
        Gt = 10 ** (p.tx_ant_gain_db / 10.0)
        Gr = 10 ** (p.rx_ant_gain_db / 10.0)

        # Friis power gain
        # Factor = (lambda / 4 * pi * d)^2
        G_friis = Gt * Gr * (self.wavelength / (4 * np.pi * p.distance_m)) ** 2
        
        # --- THE CLAMP ---
        # Gain cannot exceed the product of antenna gains (max coupling)
        G = np.minimum(G_friis, Gt * Gr)

        # Convert to amplitude scaling
        alpha = np.sqrt(G)

        y = alpha * x
        return s.copy_with(x=y)