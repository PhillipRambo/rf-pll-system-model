import numpy as np

from rfmodel.core.config import load_yaml
from rfmodel.core.signal import Signal
from rfmodel.core.pipeline_builder import pipeline_from_config

# This import is required so the registry registers block types (like "lna")
import rfmodel.rf.registry  # noqa: F401

cfg = load_yaml("configs/pipeline_LNA_demo.yaml")
pipe = pipeline_from_config(cfg)

# Create a dummy signal
fs = 20e6
N = 100000
rng = np.random.default_rng(0)
x = (rng.normal(0, 1/np.sqrt(2), N) + 1j*rng.normal(0, 1/np.sqrt(2), N)).astype(np.complex128)
sig_in = Signal(x=x, fs_hz=fs)

sig_out, taps = pipe.run(sig_in, taps=["lna1"])

print("Blocks:", [b.name for b in pipe.blocks])
print("Pin :", np.mean(np.abs(sig_in.x)**2))
print("Pout:", np.mean(np.abs(sig_out.x)**2))
