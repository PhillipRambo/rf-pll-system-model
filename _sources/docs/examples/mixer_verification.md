# Mixer — Linearity and Noise

A multi-part verification of the mixer block covering three test cases:

1. **Ideal mixer** — all impairments disabled (`mixer_ideal=True`). Verifies that signal power is preserved exactly through the block and that the input and output spectra are identical.

2. **IIP3 two-tone test** — follows the same procedure as the LNA nonlinearity test. A two-tone complex envelope is swept in power, the fundamental and IM3 powers are tracked, and linear fits are extrapolated to find the IIP3. Result is compared against the configured `iip3_dbm`.

3. **Noise figure test** — uses a thermal noise input to measure the added noise and compute the noise figure, following the same method as the LNA noise notebook.

**Key checks:**
- Ideal mixer: power ratio = 1.0, spectra identical
- Measured IIP3 matches `iip3_dbm`
- Measured NF matches `nf_db`

## Pipeline configuration

```{literalinclude} ../../test_benches/Devices/Mixer_PLL/PLL_mixer.yaml
:language: yaml
```
