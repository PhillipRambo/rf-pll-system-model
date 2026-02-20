# src/rfmodel/rf/lna.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.units import db_to_linear
from rfmodel.core.random import get_rng  # or however you manage RNG
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class LNAParams:
    gain_db: float
    nf_db: float
    temp_k: float = 290.0


class LNABlock(Block):  # Define an LNA block type that inherits common behavior from Block
    type_name = "lna"   # Class-level identifier used to label/recognize this block type

    def __init__(self, name: str, params: LNAParams, seed: int | None = None):
        super().__init__(name=name)   # Initialize the base Block part (e.g., store the instance name)
        self.params = params          # Store configuration (gain, NF, temperature)
        self._rng = get_rng(seed)     # Store per-instance RNG for noise generation (seed controls repeatability)

    def process(self, s: Signal) -> Signal:  # Process an input Signal and return a new Signal
        p = self.params

        g_lin = db_to_linear(p.gain_db)
        y = s.x * np.sqrt(g_lin)      # Apply amplitude gain corresponding to power gain g_lin

        F = db_to_linear(p.nf_db)
        k = 1.380649e-23
        N0_out_added = (F - 1.0) * k * p.temp_k * g_lin  # Added output noise PSD due to NF
        var = N0_out_added * s.fs_hz                     # Complex noise variance per sample

        n = (self._rng.normal(0.0, np.sqrt(var / 2), size=y.shape) +
             1j * self._rng.normal(0.0, np.sqrt(var / 2), size=y.shape))

        y = y + n
        return s.copy_with(x=y)       # Return a new Signal with the same metadata but updated samples

'''
------- Mapping Example -------

/// Create params  ///

params = LNAParams(gain_db=20, nf_db=3)

/// Create the LNA block Object ///

lna = LNABlock('LNA1', params, seed=0)

Constructs and LNAblock and lna.params 
and lna._rng along with lna.name now lives inside

/// Use the object to  ///

S_out = lna.process(sig)

Now the process inside runs and returns the copy of sig
as the output


-------------------------------
'''