from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.units import db_to_linear, dbm_to_w
from rfmodel.core.random import get_rng
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block

@dataclass
class PLLParams:
    """Parameters for the Local Oscillator (LO) source."""
    phase_noise_std: float = 0.001  # Standard deviation of phase steps (Wiener process)
    freq_error_hz: float = 0.0      # Constant frequency offset in Hz

class PLL:
    """
    Simulates the Local Oscillator (LO) reference signal.
    In complex baseband, this is represented as a unit-magnitude 
    phasor containing phase and frequency errors.
    """
    def __init__(self, params: PLLParams, rng):
        self.p = params
        self._rng = rng

    def generate_lo_impairment(self, num_samples: int, fs: float) -> np.ndarray:
        t = np.arange(num_samples) / fs
        
        # 1. Frequency Offset (Constant rotation)
        phase_freq = 2 * np.pi * self.p.freq_error_hz * t
        
        # 2. Phase Noise (Random walk / Jitter)
        # Represented as the integration of Gaussian noise
        phase_jitter = np.cumsum(self._rng.normal(0, self.p.phase_noise_std, size=num_samples))
        
        return np.exp(1j * (phase_freq + phase_jitter))

@dataclass
class MixerParams:
    """Parameters for the I/Q Mixer hardware."""
    gain_db: float
    iip3_dbm: float
    nf_db: float
    iq_amp_imb_db: float = 0.0     # Gain mismatch between I and Q paths
    iq_phase_imb_deg: float = 0.0   # Phase mismatch from perfect 90 degrees
    dc_offset_complex: complex = 0.0 + 0.0j
    pll: PLLParams | None = None    # Nested PLL configuration

class MixerBlock(Block):
    """
    Complex-envelope Mixer model.
    Clearly separates the PLL (LO source) from Mixer (Hardware impairments).
    """
    type_name = "mixer"

    def __init__(self, name: str, params: MixerParams, seed: int | None = None):
        super().__init__(name=name)
        self.params = params
        self._rng = get_rng(seed)
        
        # Composition: The Mixer "owns" a PLL instance
        if self.params.pll:
            self.pll = PLL(self.params.pll, self._rng)
        else:
            self.pll = None

    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x
        
        # --- Stage 1: PLL (LO Injection) ---
        # Multiplies the signal by the imperfect LO phasor
        if self.pll:
            lo_signal = self.pll.generate_lo_impairment(len(x), s.fs_hz)
            x = x * lo_signal

        # --- Stage 2: Mixer Nonlinearity (IM3) ---
        # Modeled as power-normalized polynomial distortion
        G = db_to_linear(p.gain_db)
        alpha = np.sqrt(G)
        # Relating IIP3 to the cubic coefficient
        beta = alpha * (2.0 / dbm_to_w(p.iip3_dbm))
        y = alpha * x - beta * (np.abs(x)**2) * x

        # --- Stage 3: Mixer I/Q Imbalance ---
        # Models hardware mismatch in the quadrature splitter
        g_imb = db_to_linear(p.iq_amp_imb_db)
        phi_imb = np.radians(p.iq_phase_imb_deg)
        
        I = np.real(y)
        Q = np.imag(y)
        
        # I is reference; Q experiences gain and phase deviation
        y_impaired = I + 1j * (g_imb * (Q * np.cos(phi_imb) - I * np.sin(phi_imb)))

        # --- Stage 4: DC Offset ---
        # LO Leakage or Mixer self-mixing
        y_final = y_impaired + p.dc_offset_complex

        return s.copy_with(x=y_final)