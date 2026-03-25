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


class PABlock(Block):
    """
    Simple spec-driven PA model using a Rapp-like AM-AM law.

    Parameters
    ----------
    gain_db :
        Small-signal power gain in dB.

    p1db_out_dbm :
        Output 1 dB compression point in dBm.

    smoothness_p :
        Controls knee sharpness. Larger => sharper transition.
    """

    type_name = "pa"

    def __init__(self, name: str, params: PAParams):
        super().__init__(name=name)
        self.params = params

        if params.smoothness_p <= 0:
            raise ValueError("smoothness_p must be > 0")

        self.G = db_to_linear(params.gain_db)      # power gain
        self.g = np.sqrt(self.G)                   # amplitude gain
        self.p = params.smoothness_p

        # Actual output power at 1 dB compression
        P1dB_out_w = dbm_to_w(params.p1db_out_dbm)
        r_out_1dB = np.sqrt(P1dB_out_w)

        # 1 dB compression in amplitude
        c = 10.0 ** (-1.0 / 20.0)

        # Find linear output amplitude that would be 1 dB above actual
        r_lin_1dB = r_out_1dB / c

        # Solve Rapp parameter Asat so that compression is exactly 1 dB there
        # r_out = r_lin / (1 + (r_lin / Asat)^(2p))^(1/(2p))
        # => c = 1 / (1 + (r_lin / Asat)^(2p))^(1/(2p))
        # => Asat = r_lin / (c^(-2p) - 1)^(1/(2p))
        self.Asat = r_lin_1dB / ((c ** (-2.0 * self.p) - 1.0) ** (1.0 / (2.0 * self.p)))

    def process(self, s: Signal) -> Signal:
        x = s.x
        r = np.abs(x)

        # Linear output amplitude before compression
        r_lin = self.g * r

        # Rapp AM-AM compression
        r_out = r_lin / (1.0 + (r_lin / self.Asat) ** (2.0 * self.p)) ** (1.0 / (2.0 * self.p))

        gain_amp = np.zeros_like(r_out)
        nz = r > 0
        gain_amp[nz] = r_out[nz] / r[nz]

        y = gain_amp * x
        return s.copy_with(x=y)