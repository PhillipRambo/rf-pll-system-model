from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.units import db_to_linear, dbm_to_w
from rfmodel.core.random import get_rng
from rfmodel.core.signal import Signal
from rfmodel.core.block import Block

@dataclass
class PLLParams:
    VCO_Phase_Noise_dBc: tuple[float, float] # Takes phase noise in dBc at offset frequency [PN,f_offset]
    SLF_dBc: float #Low frequency noise floor
    f_L: float #Loop Bandwidth
    Tu: float  # OFDM usefull length of symbol length, i.e length of FFT interval
    enable_ofdm_weighting: bool = False #flag to enable OFDM weighting function
    f_range_limits: tuple[float, float] = (10, 1e10) # offset frequencies to evaluate the Phase noise over

class PLL:
    def __init__(self, params: PLLParams, rng):
        self.p = params
        self._rng = rng
        self.alpha = 10**(float(self.p.VCO_Phase_Noise_dBc[0]) / 10) * (float(self.p.VCO_Phase_Noise_dBc[1]))**2
        self.SLF = 10**(self.p.SLF_dBc / 10)

    def get_psd(self, f: np.ndarray) -> np.ndarray:
        f_L = self.p.f_L
        
        # Avoid division by zero at f=0
        f_safe = np.where(f == 0, np.finfo(float).eps, f)
        
        lp_factor = 1 / (1 + (f_safe / f_L)**2)   # low-pass: reference noise
        hp_factor = (f_safe / f_L)**2 / (1 + (f_safe / f_L)**2)  # high-pass: VCO noise
        
        S_phi = self.SLF * lp_factor + (self.alpha / f_safe**2) * hp_factor
        
        if self.p.enable_ofdm_weighting:
            denom = (np.pi * f_safe * self.p.Tu)**2
            h_bbf = 1 - np.divide(
                np.sin(np.pi * f_safe * self.p.Tu)**2,
                denom,
                out=np.zeros_like(f_safe),
                where=denom != 0
            )
            S_phi *= h_bbf
            
        return S_phi

    def generate_lo_impairment(self, N: int, fs: float) -> np.ndarray:
        """Generates a time-domain phasor e^(j*phi(t)) with modeled phase noise."""
        df = fs / N
        f = np.fft.rfftfreq(N, 1/fs)
        
        #Get the PSD for these specific frequencies
        S_phi = self.get_psd(f)

        #Convert PSD to frequency-domain noise (amplitude scaling)
        phi_f = (self._rng.standard_normal(len(f)) + 1j * self._rng.standard_normal(len(f)))
        phi_f *= np.sqrt(S_phi * df) * (N / 2)
        phi_f[0] = 0.0  # zero DC: a constant phase offset has no physical meaning

        phi_t = np.fft.irfft(phi_f, n=N)
        
        #Return the LO phasor
        return np.exp(1j * phi_t)

@dataclass
class MixerParams:
    gain_db: float
    iip3_dbm: float
    nf_db: float
    iq_amp_imb_db: float = 0.0
    iq_phase_imb_deg: float = 0.0
    dc_offset_complex: complex = 0.0 + 0.0j
    pll: PLLParams | None = None
    # Flag for "Ideal Mixer" mode
    mixer_ideal: bool = False 

class MixerBlock(Block):
    type_name = "mixer"

    def __init__(self, name: str, params: MixerParams, seed: int | None = None):
        super().__init__(name=name)
        self.params = params
        self._rng = get_rng(seed)
        if self.params.pll is not None:
            self.pll = PLL(self.params.pll, self._rng)
        else:
            self.pll = None
    def process(self, s: Signal) -> Signal:
        p = self.params
        x = s.x
        
        # --- Stage 1: PLL Phase Noise (Always applied if PLL exists) ---
        if self.pll:
            lo_signal = self.pll.generate_lo_impairment(len(x), s.fs_hz)
            x = x * lo_signal

        # --- Check for Ideal Mixer Flag ---
        if p.mixer_ideal:
            # Skip Stages 2-4 and just return signal with PLL noise
            return s.copy_with(x=x)

        # --- Stage 2: Nonlinearity ---
        G = db_to_linear(p.gain_db)
        alpha_lin = np.sqrt(G)
        beta = alpha_lin * (2.0 / dbm_to_w(p.iip3_dbm))
        y = alpha_lin * x - beta * (np.abs(x)**2) * x

        # --- Stage 3: I/Q Imbalance ---
        g_imb = db_to_linear(p.iq_amp_imb_db)
        phi_imb = np.radians(p.iq_phase_imb_deg)
        I, Q = np.real(y), np.imag(y)
        y_impaired = I + 1j * (g_imb * (Q * np.cos(phi_imb) - I * np.sin(phi_imb)))

        # --- Stage 4: DC Offset ---
        y_final = y_impaired + p.dc_offset_complex

        return s.copy_with(x=y_final)