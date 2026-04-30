# System Performance Verification

The top-level end-to-end system verification. The full signal chain — payload generation, RF front-end (LNA, Mixer with PLL, PA), channel (path loss, AWGN), and demodulation — is simulated and evaluated against performance targets.

Key metrics:
- **EVM (Error Vector Magnitude)** — measures the deviation of received constellation points from their ideal positions, combining all impairments from the RF chain and channel
- **BER (Bit Error Rate)** — end-to-end bit errors after demapping

This is the primary notebook for assessing the overall impact of RF impairments on system performance. Results can be compared against the specification targets in the project report.

## Pipeline configuration

```{literalinclude} ../../verification/Tx_channel_Rx.yaml
:language: yaml
```
