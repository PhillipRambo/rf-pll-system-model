# src/rfmodel/rf/lna.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.units import db_to_linear, dbm_to_w
from rfmodel.core.random import get_rng  # or however you manage RNG
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class LNAParams:
    gain_db: float
    nf_db: float
    IP3_dbm: float
    R_in: float
    R_out: float
    temp_k: float = 290.0


class LNABlock(Block):
    """
    Complex-envelope LNA behavioral model (power-normalized).

    Signal convention
    -----------------
    - x is a complex envelope (analytic signal)

    Parameters
    ----------
    - gain_db : power gain [dB]
    - IP3_dbm : input-referred IIP3 [dBm]
    - nf_db   : noise figure [dB]
    - temp_k  : noise temperature [K]
    """

    type_name = "lna"

    def __init__(self, name: str, params: LNAParams, seed: int | None = None):
        super().__init__(name=name)
        self.params = params
        self._rng = get_rng(seed)

    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x

        # ---- Linear gain (power gain) ----
        # Power gain G maps to complex-envelope amplitude gain alpha = sqrt(G).
        G = db_to_linear(p.gain_db)          # power gain (linear)
        alpha = np.sqrt(G)                  # amplitude gain
        y = alpha * x

        # ---- 3rd-order nonlinearity from IIP3 (complex-envelope form) ----
        # Model:
        #   y = alpha*x + beta*x*|x|^2
        #
        # For the standard two-tone IIP3 definition (input-referred):
        #   P_IIP3 = (4/3) * alpha / |beta|
        # => |beta| = (4/3) * alpha / P_IIP3
        #
        # Choose beta negative to produce gain compression.
        P_iip3_w = dbm_to_w(p.IP3_dbm)       # [W]
        beta_mag = (4.0 / 3.0) * alpha / P_iip3_w
        beta = -beta_mag
        y = y + beta * x * (np.abs(x) ** 2)  # distortion uses input x (input-referred IP3)

        # ---- Added output noise from NF ----
        # Added output noise PSD (one-sided, per the usual kT*B convention):
        #   N0_out_added = (F - 1) * k * T * G   [W/Hz]
        F = db_to_linear(p.nf_db)
        k = 1.380649e-23
        noise_psd_w_per_hz = (F - 1.0) * k * p.temp_k * G

        # Complex envelope sampled at fs represents RF bandwidth B = fs/2.
        B_hz = s.fs_hz / 2.0
        Pn_out_added_w = noise_psd_w_per_hz * B_hz  # [W] = E[|n|^2] per sample (white over band)

        # Proper complex Gaussian: E[|n|^2] = 2*sigma^2  => sigma = sqrt(P/2)
        sigma = np.sqrt(Pn_out_added_w / 2.0) # remembering that its splitted over I/Q
        n = (self._rng.normal(0.0, sigma, size=y.shape) +
             1j * self._rng.normal(0.0, sigma, size=y.shape))

        y = y + n
        return s.copy_with(x=y)


