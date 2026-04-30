# PA — Compression and Model Comparison

Tests the PA block across a range of input powers to characterise the AM-AM compression curve. The notebook covers three things:

1. **Small-signal gain check** — verifies that at low input power the measured gain matches `gain_db`
2. **Compression sweep** — sweeps input power from −30 to +20 dBm, measures output power and gain at each point, and locates the 1 dB compression point
3. **Rapp vs cubic model comparison** — runs the same sweep with both models and overlays the results, illustrating the difference in compression shape and the unphysical gain reversal of the cubic model well above P1dB

The Rapp model is the recommended default; the cubic model is available for analytical work near the compression point.

**Key checks:**
- Small-signal gain matches `gain_db`
- Measured output P1dB matches configured `p1db_out_dbm`
- Rapp model saturates gracefully; cubic model diverges above P1dB

## Pipeline configuration

```{literalinclude} ../../test_benches/Devices/PA/PA_test.yaml
:language: yaml
```
