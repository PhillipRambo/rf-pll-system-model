from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.block import Block
from rfmodel.core.signal import Signal
from rfmodel.core.units import db_to_linear, dbm_to_w


@dataclass
class PAParams:
    gain_db: float
    p1db_out_dbm: float
    smoothness_p: float = 2.0
    enable_cubic: bool = False


class PABlock(Block):
    """
    Spec-driven PA model with two selectable AM-AM laws:
      - Rapp-like soft compression (default)
      - Memoryless cubic: y = alpha*x - beta*|x|^2 * x

    Signal convention
    -----------------
    Power-normalized complex envelope: |x|^2 is instantaneous power in W.
    alpha = sqrt(G) is dimensionless, beta has units [1/W].

    Parameters
    ----------
    gain_db :
        Small-signal power gain in dB.
    p1db_out_dbm :
        Output 1 dB compression point in dBm.
    smoothness_p :
        Rapp knee sharpness. Larger => sharper transition.
    enable_cubic :
        If True, use the cubic AM-AM law instead of Rapp.
    """

    type_name = "pa"

    def __init__(self, name: str, params: PAParams):
        super().__init__(name=name)
        self.params = params

        if params.smoothness_p <= 0:
            raise ValueError("smoothness_p must be > 0")

        self.G = db_to_linear(params.gain_db)
        self.alpha = np.sqrt(self.G)
        self.g = self.alpha
        self.p = params.smoothness_p

        P1dB_out_w = dbm_to_w(params.p1db_out_dbm)
        r_out_1dB = np.sqrt(P1dB_out_w)
        c = 10.0 ** (-1.0 / 20.0)

        # Rapp: solve Asat so compression is exactly 1 dB at P1dB_out
        r_lin_1dB = r_out_1dB / c
        self.Asat = r_lin_1dB / ((c ** (-2.0 * self.p) - 1.0) ** (1.0 / (2.0 * self.p)))

        self.beta_cubic = (1-c) * c**2 * ( self.alpha**3 / (P1dB_out_w) )

    def process(self, s: Signal) -> Signal:
        x = s.x

        if self.params.enable_cubic:
            y = self.alpha * x - self.beta_cubic * (np.abs(x) ** 2) * x
            return s.copy_with(x=y)

        r = np.abs(x)
        r_lin = self.g * r
        r_out = r_lin / (1.0 + (r_lin / self.Asat) ** (2.0 * self.p)) ** (1.0 / (2.0 * self.p))

        gain_amp = np.zeros_like(r_out)
        nz = r > 0
        gain_amp[nz] = r_out[nz] / r[nz]

        y = gain_amp * x
        return s.copy_with(x=y)
