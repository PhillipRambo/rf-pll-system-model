import math
from typing import Union

Number = Union[float, int]

def db_to_linear(db: Number) -> float:
    """Convert dB to linear ratio."""
    return 10 ** (float(db) / 10.0)

def linear_to_db(x: Number) -> float:
    """Convert linear ratio to dB."""
    x = float(x)
    if x <= 0:
        raise ValueError("Linear value must be > 0")
    return 10.0 * math.log10(x)

def w_to_dbm(p_w: Number) -> float:
    """Convert Watts to dBm."""
    p_w = float(p_w)
    if p_w <= 0:
        raise ValueError("Power in W must be > 0")
    return 10.0 * math.log10(p_w / 1e-3)

def dbm_to_w(p_dbm: Number) -> float:
    """Convert dBm to Watts."""
    return 1e-3 * 10 ** (float(p_dbm) / 10.0)

def thermal_noise_density_dbm_hz(temp_k: float = 290.0) -> float:
    """
    Thermal noise power density in dBm/Hz.
    At 290 K this is about -174 dBm/Hz.
    """
    if temp_k <= 0:
        raise ValueError("temp_k must be > 0")
    return -174.0 + 10.0 * math.log10(temp_k / 290.0)

def kTB_dbm(bw_hz: Number, temp_k: float = 290.0) -> float:
    """
    Thermal noise power over bandwidth BW (Hz) in dBm.
    """
    bw_hz = float(bw_hz)
    if bw_hz <= 0:
        raise ValueError("bw_hz must be > 0")
    return thermal_noise_density_dbm_hz(temp_k) + 10.0 * math.log10(bw_hz)
