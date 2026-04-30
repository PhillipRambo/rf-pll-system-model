# Core Framework

The core framework provides the building blocks for constructing signal processing pipelines. It is designed to be modular, composable, and reproducible. This chapter covers the key abstractions: `Signal`, `Block`, `Pipeline`, the YAML configuration system, and supporting utilities.

---

## Signal

The `Signal` class is the fundamental data container that flows through the entire pipeline. It is defined as a **frozen dataclass** — once created, its fields cannot be modified. This immutability is intentional: it prevents accidental in-place mutation and encourages a functional, composable style where each block returns a new `Signal` rather than modifying an existing one.

### Fields

| Field | Type | Description |
|---|---|---|
| `x` | `np.ndarray` (complex) | Complex baseband samples. `\|x\|^2` is instantaneous power in watts. |
| `fs_hz` | `float` | Sampling rate in Hz |
| `fc_hz` | `float` | Centre frequency in Hz (optional, defaults to 0) |
| `meta` | `dict` | Arbitrary metadata (modulation order, labels, etc.) |

### Creating and copying signals

```python
from rfmodel.core.signal import Signal
import numpy as np

sig = Signal(x=np.zeros(1024, dtype=complex), fs_hz=10e6, fc_hz=2.4e9, meta={})

# Create a modified copy (all other fields unchanged)
sig2 = sig.copy_with(x=new_samples, meta={"gain_db": 20})
```

The `copy_with()` method is how blocks return their output — they receive a `Signal`, process `x`, and return `sig.copy_with(x=y)`. This pattern preserves all metadata and configuration from the input signal unless explicitly overridden.

### Signal convention

The package uses the **power-normalised complex envelope** convention throughout:

- $x[n]$ is a complex baseband sample
- $|x[n]|^2$ is the instantaneous power in watts
- For a single tone of amplitude $A$: $x[n] = A e^{j\omega n}$, average power = $A^2$

This means voltage-domain thinking (normalised to 1 Ω) applies everywhere. Conversions to dBm use $P_\text{dBm} = 10 \log_{10}(P_W / 10^{-3})$.

---

## Block

`Block` is the abstract base class for all signal processing units. Every component in the package — LNA, PA, Mixer, AWGN, QAM modulator — is a `Block`.

### Interface

```python
from rfmodel.core.block import Block
from rfmodel.core.signal import Signal

class MyBlock(Block):
    type_name = "my_block"

    def process(self, s: Signal) -> Signal:
        # transform s.x, return new Signal
        return s.copy_with(x=s.x * 2)
```

### Key behaviour

- **`process(s)`** — the core transform. Must be implemented by every subclass.
- **`__call__(s)`** — wraps `process()`. If `self.enabled = False`, the signal passes through unchanged (bypass mode).
- **`reset(seed=None)`** — reinitialises internal state (RNG, filter memory). Called automatically by `Pipeline.reset()`.
- **`enabled`** — boolean flag. Set via `pipeline.enable("block_name", False)` to bypass a block without removing it.

### Bypass pattern

The `enabled` flag is useful for A/B comparisons:

```python
pipe.enable("lna1", False)   # bypass the LNA
out_no_lna, _ = pipe.run(sig_in)

pipe.enable("lna1", True)    # restore
out_with_lna, _ = pipe.run(sig_in)
```

---

## Pipeline

`Pipeline` holds an ordered list of `Block` objects and runs a `Signal` through them sequentially. It is the primary way to compose a system model.

```
sig_in → [Block A] → [Block B] → [Block C] → sig_out
```

### Running a pipeline

```python
from rfmodel.core.pipeline import Pipeline

pipe = Pipeline()
pipe.add(lna_block)
pipe.add(mixer_block)
pipe.add(pa_block)

sig_out, taps = pipe.run(sig_in)
```

### Tapping intermediate signals

Any block output can be captured without modifying the pipeline:

```python
sig_out, taps = pipe.run(sig_in, taps=["lna1", "mixer1"])

# Inspect signal at LNA output
lna_output = taps["lna1"]
print("LNA output power:", np.mean(np.abs(lna_output.x)**2), "W")
```

This is particularly useful for debugging gain budgets and noise figures at intermediate stages.

### Modifying a pipeline at runtime

The pipeline supports full runtime manipulation:

```python
# Insert a block after an existing one
pipe.add(new_filter, after="lna1")

# Replace a block (keeps its position)
pipe.replace("lna1", LNABlock("lna1", new_params))

# Remove a block
pipe.remove("lna1")

# Toggle a block on/off
pipe.enable("lna1", False)

# Access a block directly to modify its parameters
lna = pipe.get("lna1")
lna.params.gain_db = 15.0
```

This flexibility makes it straightforward to sweep parameters, compare models, or study individual impairments in isolation.

### Resetting state

Blocks that contain internal state (e.g., RNG for noise generation) can be reset deterministically:

```python
pipe.reset(seed=42)   # reproducible noise realisations
sig_out_1, _ = pipe.run(sig_in)

pipe.reset(seed=42)   # same seed → identical noise
sig_out_2, _ = pipe.run(sig_in)
```

---

## YAML Configuration

Pipelines can be fully specified in YAML and loaded with a single call. This separates system configuration from code, making it easy to store, version, and reuse setups.

### Example YAML

```yaml
pipeline:
  - type: lna
    name: lna1
    seed: 1
    params:
      gain_db: 20.0
      nf_db: 3.0
      IP3_dbm: 10.0

  - type: mixer
    name: mixer1
    seed: 2
    params:
      gain_db: 0.0
      iip3_dbm: 20.0
      nf_db: 8.0
      pll:
        VCO_PhaseNoise: [-90, 1.0e6]
        LF_noise_floor: -140
        loop_bandwidth: 10.0e3

  - type: pa
    name: pa1
    params:
      gain_db: 30.0
      p1db_out_dbm: 27.0
      smoothness_p: 5.0
```

### Loading and running

```python
from rfmodel.core.config import load_yaml
from rfmodel.core.pipeline_builder import pipeline_from_config

cfg  = load_yaml("path/to/config.yaml")
pipe = pipeline_from_config(cfg)

sig_out, _ = pipe.run(sig_in)
```

The builder reads the `pipeline` list, looks up each `type` in the registry, and instantiates the corresponding block with the given parameters and seed. The result is a fully-wired `Pipeline` ready to run.

---

## Factory and Registry Pattern

The package uses a decorator-based registry to decouple YAML keys from block constructors. Each module (rf, channel, comms) registers its builders independently.

```python
from rfmodel.core.factory import register_block

@register_block("lna")
def _build_lna(cfg: dict) -> LNABlock:
    p = cfg["params"]
    return LNABlock(
        name=cfg["name"],
        params=LNAParams(
            gain_db=p["gain_db"],
            nf_db=p["nf_db"],
            IP3_dbm=p["IP3_dbm"],
        ),
        seed=cfg.get("seed"),
    )
```

To add a new block type to the system, simply define a builder function, decorate it with `@register_block("your_type")`, and import its module. The YAML loader will then recognise `type: your_type` automatically.

---

## Units Utilities

All RF engineering unit conversions are centralised in `rfmodel.core.units`:

```python
from rfmodel.core.units import db_to_linear, dbm_to_w, w_to_dbm, kTB_dbm

G = db_to_linear(20.0)         # 100.0
P_w = dbm_to_w(10.0)           # 0.01 W
P_dbm = w_to_dbm(0.01)         # 10.0 dBm
noise_floor = kTB_dbm(bw_hz=10e6, temp_k=290)  # thermal noise power in dBm
```

| Function | Description |
|---|---|
| `db_to_linear(x)` | $10^{x/10}$ |
| `linear_to_db(x)` | $10 \log_{10}(x)$ |
| `dbm_to_w(x)` | $10^{(x-30)/10}$ |
| `w_to_dbm(x)` | $10 \log_{10}(x) + 30$ |
| `thermal_noise_density_dbm_hz(T)` | $10\log_{10}(kT) + 30$ in dBm/Hz |
| `kTB_dbm(bw_hz, T)` | Total thermal noise power $kTB$ in dBm |

---

## RNG Management

All stochastic blocks (noise sources, PLL) accept an optional `seed` parameter for reproducibility. Internally they use `numpy.random.Generator` via `rfmodel.core.random.get_rng()`.

```python
lna = LNABlock("lna1", params, seed=42)  # fixed seed → reproducible noise
lna = LNABlock("lna1", params, seed=None) # unseeded → different noise each run
```

Seeding is particularly important when comparing two configurations: use the same seed on both so differences in output come from the parameter change, not from noise variation.
