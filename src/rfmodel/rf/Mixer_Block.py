from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.units import db_to_linear, dbm_to_w
from rfmodel.core.random import get_rng
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block

@dataclass
class PLLParams:
    """PLL phase noise defined via a target at 1 MHz offset."""
    L_1MHz_dBc: float          # Phase noise at 1 MHz offset (dBc/Hz)
    f_L: float                 # Loop bandwidth (Hz)
    S_LF: float                # Low-frequency noise floor (linear, rad^2/Hz)
    freq_error_hz: float = 0.0

class PLL:
    """
    Phase noise generated using Lorentzian PSD model instead of Wiener process.
    """
    def __init__(self, params: PLLParams, rng):
        self.p = params
        self._rng = rng

        # Convert 1 MHz PN spec to alpha
        S_1MHz = 10**(self.p.L_1MHz_dBc / 10)
        self.alpha = S_1MHz * (1e6)**2

    def _generate_phase_noise(self, num_samples: int, fs: float) -> np.ndarray:
        """
        Generate phase noise by filtering white noise to match Lorentzian PSD.
        Implements first-order OU process equivalent.
        """
        dt = 1.0 / fs
        f_L = self.p.f_L

        # Discrete-time OU process coefficient
        a = np.exp(-2 * np.pi * f_L * dt)

        # Noise scaling (derived from continuous PSD)
        sigma_w = np.sqrt((1 - a**2) * self.alpha / (2 * f_L))

        w = self._rng.normal(0, sigma_w, size=num_samples)
        phi = np.zeros(num_samples)

        for n in range(1, num_samples):
            phi[n] = a * phi[n-1] + w[n]

        return phi

    def generate_lo_impairment(self, num_samples: int, fs: float) -> np.ndarray:
        t = np.arange(num_samples) / fs

        # Frequency offset
        phase_freq = 2 * np.pi * self.p.freq_error_hz * t

        # Phase noise from Lorentzian model
        phase_noise = self._generate_phase_noise(num_samples, fs)

        return np.exp(1j * (phase_freq + phase_noise))
    

    
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