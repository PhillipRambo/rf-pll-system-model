import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.pipeline import Pipeline
from rfmodel.rf.LNA import LNABlock, LNAParams


def _complex_gaussian(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # E[|x|^2] ≈ 1
    return (rng.normal(0.0, 1/np.sqrt(2), N) + 1j*rng.normal(0.0, 1/np.sqrt(2), N)).astype(np.complex128)


def test_pipeline_runs_and_gain_nf0():
    fs = 20e6
    N = 200000
    x = _complex_gaussian(N, seed=0)
    sig_in = Signal(x=x, fs_hz=fs)

    pipe = Pipeline()
    pipe.add(LNABlock("lna", LNAParams(gain_db=20.0, nf_db=0.0), seed=123))

    sig_out, taps = pipe.run(sig_in, taps=["lna"])

    pin = np.mean(np.abs(sig_in.x) ** 2)
    pout = np.mean(np.abs(sig_out.x) ** 2)

    gain_db_meas = 10 * np.log10(pout / pin)

    # With NF=0, gain should be close to 20 dB (allow small numerical tolerance)
    assert abs(gain_db_meas - 20.0) < 0.2
    assert "lna" in taps


def test_pipeline_noise_increases_power():
    fs = 20e6
    N = 200000
    x = _complex_gaussian(N, seed=0)
    sig_in = Signal(x=x, fs_hz=fs)

    pipe_nf0 = Pipeline()
    pipe_nf0.add(LNABlock("lna", LNAParams(gain_db=20.0, nf_db=0.0), seed=123))
    out0, _ = pipe_nf0.run(sig_in)

    pipe_nf6 = Pipeline()
    pipe_nf6.add(LNABlock("lna", LNAParams(gain_db=20.0, nf_db=6.0), seed=123))
    out6, _ = pipe_nf6.run(sig_in)

    p0 = np.mean(np.abs(out0.x) ** 2)
    p6 = np.mean(np.abs(out6.x) ** 2)

    # Added noise should increase total output power
    assert p6 > p0


def test_pipeline_reproducible_with_seed():
    fs = 20e6
    N = 200000
    x = _complex_gaussian(N, seed=0)
    sig_in = Signal(x=x, fs_hz=fs)

    pipe1 = Pipeline()
    pipe1.add(LNABlock("lna", LNAParams(gain_db=20.0, nf_db=6.0), seed=999))
    out1, _ = pipe1.run(sig_in)

    pipe2 = Pipeline()
    pipe2.add(LNABlock("lna", LNAParams(gain_db=20.0, nf_db=6.0), seed=999))
    out2, _ = pipe2.run(sig_in)

    # Same seed => identical noise => identical output
    assert np.allclose(out1.x, out2.x)
