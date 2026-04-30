# PLL — Phase Noise Verification

Verifies the PLL phase noise model by generating an LO signal from the configured PLL and measuring its phase noise spectrum. The phase noise curve is extracted using the phase noise analyser (Welch method on the unwrapped instantaneous phase) and plotted against the model.

The two-region PLL noise shape is clearly visible: inside the loop bandwidth $f_L$, the spectrum is dominated by flat reference noise; outside $f_L$, the free-running VCO $1/f^2$ Lorentzian takes over. The crossover at the loop bandwidth is identifiable in the measured PSD.

**Key checks:**
- Phase noise at the reference offset matches `VCO_Phase_Noise_dBc`
- Loop bandwidth knee visible at `f_L`
- Low-frequency noise floor matches `SLF_dBc`

## Pipeline configuration

```{literalinclude} ../../test_benches/Devices/Mixer_PLL/PLL_mixer.yaml
:language: yaml
```
