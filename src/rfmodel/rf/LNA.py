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
        
        # 1. Linear gain
        G = db_to_linear(p.gain_db)
        alpha = np.sqrt(G)

        # 2. Correct Nonlinearity: x * |x|^2
        # Relate IIP3 to the cubic coefficient alpha_3 (complex)
        # Standard: |alpha_3 / alpha| = 2 / P_iip3 (linear power)
        Pin_iip3_w = dbm_to_w(p.IP3_dbm)
        beta = alpha * (2.0 / Pin_iip3_w)
        
        # Note: beta needs to be complex if you want to model AM-PM distortion, 
        # otherwise real if just gain compression.
        y = alpha * x - beta * (np.abs(x)**2) * x

        # ---- Added output noise from NF ----
        # Added output noise PSD (one-sided): N0_out_added = (F - 1) * k * T * G  [W/Hz]
        F = db_to_linear(p.nf_db)
        k = 1.380649e-23
        noise_psd_w_per_hz = (F - 1.0) * k * p.temp_k * G

        # Complex envelope sampled at fs represents RF bandwidth B = fs/2.
        B_hz = s.fs_hz / 2.0
        Pn_out_added_w = noise_psd_w_per_hz * B_hz  # [W] = E[|n|^2] per sample

        # Proper complex Gaussian: E[|n|^2] = 2*sigma^2  => sigma = sqrt(P/2)
        sigma = np.sqrt(Pn_out_added_w / 2.0)
        n = (
            self._rng.normal(0.0, sigma, size=y.shape)
            + 1j * self._rng.normal(0.0, sigma, size=y.shape)
        )

        y = y + n
        return s.copy_with(x=y)
    

   