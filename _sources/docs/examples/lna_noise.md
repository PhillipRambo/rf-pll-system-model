# LNA — Noise Figure

Verifies the LNA noise model using a thermal noise input. A complex Gaussian noise source at $T = 290$ K with total power $kTB$ is generated and passed through the LNA. The output noise power is measured and the noise figure is computed as:

$$F = \frac{P_{n,\text{out}}}{G \cdot kTB}$$

The measured noise figure is compared against the configured `nf_db` parameter. The input and output noise power spectral densities are plotted to verify the flat noise floor and the noise elevation introduced by the amplifier.

**Key checks:**
- Measured NF matches configured `nf_db`
- Output noise PSD is uniformly elevated above input noise floor by the noise factor

## Pipeline configuration

```{literalinclude} ../../test_benches/Devices/LNA/pipeline_LNA_demo.yaml
:language: yaml
```
