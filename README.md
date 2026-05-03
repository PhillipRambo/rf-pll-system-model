# rf-system-model

A Python package for behavioural simulation of RF signal chains with integrated PLL phase noise modelling. Designed for end-to-end analysis of wireless communication systems at the complex baseband level — from RF front-end impairments to EVM and BER.

## What it does

- **RF front-end** — LNA, PA (Rapp and cubic models), Mixer with configurable gain, nonlinearity, and noise figure
- **PLL phase noise** — two-region model (reference + VCO noise) shaped by the loop filter, with optional OFDM weighting
- **Channel** — AWGN and free-space path loss (Friis equation)
- **Communications** — PRBS bit generation, Gray-coded QAM (4–256-QAM), OFDM modulation/demodulation
- **Measurement** — spectrum analysis, phase noise curves, EVM, BER
- **Flexible pipelines** — build signal chains from YAML or Python; tap, swap, or disable blocks at runtime

## Installation

Requires Python >= 3.10.

```bash
git clone [<repo-url>](https://github.com/PhillipRambo/rf-system-model.git)
cd rf-system-model
pip install -e .
```

Dependencies: `numpy`, `matplotlib`, `pyyaml`, `scipy`

## Quick start

```python
from rfmodel.core.config import load_yaml
from rfmodel.core.pipeline_builder import pipeline_from_config
from rfmodel.core.signal import Signal
import numpy as np

cfg  = load_yaml("configs/my_chain.yaml")
pipe = pipeline_from_config(cfg)

fs = 10e6
t  = np.arange(2048) / fs
x  = np.exp(1j * 2 * np.pi * 100e3 * t)

sig_in  = Signal(x=x, fs_hz=fs, fc_hz=2.4e9, meta={})
sig_out, _ = pipe.run(sig_in)
```

## Documentation

Full documentation — including the core framework, all components, and annotated example notebooks — is available at:

**[https://philliprambo.github.io/rf-pll-system-model/](https://philliprambo.github.io/rf-pll-system-model/)**

To build the docs locally:

```bash
pip install jupyter-book==0.15.1
jupyter-book build .
# open _build/html/index.html
```

## Running tests

```bash
pytest src/rfmodel/test/
```

## Contact

Phillip Rambo — phillipfbp@gmail.com
