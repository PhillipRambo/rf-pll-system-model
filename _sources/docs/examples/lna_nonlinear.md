# LNA — Nonlinearity

Demonstrates the IIP3 measurement procedure for the LNA using the standard two-tone method. A real two-tone RF signal is constructed and converted to the complex baseband via the Hilbert transform. The input power is swept over a wide dynamic range using a scale-factor approach, and the spectrum analyser tracks both the fundamental tone power and the third-order intermodulation (IM3) product power at each sweep point.

The resulting curves (fundamental: slope ~1 dB/dB, IM3: slope ~3 dB/dB) are fitted in the linear small-signal regime and extrapolated to their intersection, giving the input-referred IIP3. The measured intercept point is compared against the configured `IP3_dbm` parameter and the error is reported.

**Key checks:**
- Fundamental slope ≈ 1 dB/dB
- IM3 slope ≈ 3 dB/dB
- Measured IIP3 matches configured `IP3_dbm`

## Pipeline configuration

```{literalinclude} ../../test_benches/Devices/LNA/pipeline_LNA_demo.yaml
:language: yaml
```
