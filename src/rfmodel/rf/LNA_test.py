import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.rf.LNA import LNABlock, LNAParams

fs = 20e6
N = 200000

# Input: unit-power complex Gaussian (E[|x|^2] ≈ 1)
rng = np.random.default_rng(0)
x = (rng.normal(0, 1/np.sqrt(2), N) + 1j*rng.normal(0, 1/np.sqrt(2), N)).astype(np.complex128)

sig_in = Signal(x=x, fs_hz=fs)

lna = LNABlock("lna1", LNAParams(gain_db=20.0, nf_db=3.0), seed=1)
sig_out = lna.process(sig_in)

pin = np.mean(np.abs(sig_in.x)**2)
pout = np.mean(np.abs(sig_out.x)**2)

print("Pin  (mean |x|^2):", pin)
print("Pout (mean |x|^2):", pout)
print("Measured gain (dB):", 10*np.log10(pout/pin))


lna_nf0 = LNABlock("lna_nf0", LNAParams(gain_db=20.0, nf_db=0.0), seed=1)
out_nf0 = lna_nf0.process(sig_in)

lna_nf20 = LNABlock("lna_nf20", LNAParams(gain_db=20.0, nf_db=20.0), seed=1)
out_nf20 = lna_nf20.process(sig_in)

p_nf0 = np.mean(np.abs(out_nf0.x)**2)
p_nf20 = np.mean(np.abs(out_nf20.x)**2)

print("Pout NF=0 dB :", p_nf0)
print("Pout NF=20 dB:", p_nf20)
print("Extra power ratio (dB):", 10*np.log10(p_nf20/p_nf0))

