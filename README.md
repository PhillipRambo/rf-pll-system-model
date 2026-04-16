# rf-pll-system-model

A Python framework for RF system and PLL modeling, including signal chain simulation, phase noise analysis, and OFDM/QAM metrics.

## Structure

```
src/rfmodel/
├── core/         # Base abstractions: Block, Signal, Pipeline, RNG, units
├── rf/           # RF components: LNA, PA, Mixer (with PLL phase noise)
├── channel/      # Channel models: AWGN, path loss, composite channel
├── comms/        # Communications blocks: OFDM, QAM modulator, PRNG
├── meas/         # Measurement tools: spectrum analyser, phase noise analyser
├── plot_utils/   # Plotting utilities for spectra, constellations, block diagrams
├── system/       # Link budget utilities
└── test/         # Unit tests
```

Verification notebooks are in `test_benches/` and `verification/`. Configuration files (YAML) live in `configs/`.

## Install

```bash
pip install -e .
```

Requires Python >= 3.10. Dependencies: `numpy`, `matplotlib`, `pyyaml`.

## Running tests

```bash
pytest src/rfmodel/test/
```
