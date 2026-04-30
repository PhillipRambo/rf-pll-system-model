# OFDM and QAM

End-to-end test of the communications chain without RF impairments:

- PRBS bit generation
- QAM modulation and constellation visualisation
- OFDM modulation: FFT spectrum, guard bands, cyclic prefix
- OFDM demodulation and QAM demapping
- BER computation for a given SNR

This notebook establishes the baseline signal chain and verifies that the modulator/demodulator pair is error-free in an ideal (noiseless) scenario. It also shows the OFDM spectrum and the effect of the cyclic prefix on inter-symbol interference robustness.
