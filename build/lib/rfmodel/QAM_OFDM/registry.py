from __future__ import annotations
import numpy as np

from rfmodel.core.factory import register_block
from rfmodel.QAM_OFDM.OFDM_block import OFDMParams, OFDMModulator
from rfmodel.QAM_OFDM.QAM_modulator import QAMModulator, QAMParams


@register_block("qam_modulator")
def _build_qam_modulator(cfg: dict) -> QAMModulator:
    name = cfg["name"]
    p = cfg.get("params", {})

    params = QAMParams(
        M=int(p.get("M", 64)),
        gray_map=bool(p.get("gray_map", True)),
        unit_average_power=bool(p.get("unit_average_power", True)),
    )
    return QAMModulator(name=name, params=params)


@register_block("ofdm_modulator")
def _build_ofdm_modulator(cfg: dict) -> OFDMModulator:
    name = cfg["name"]
    p = cfg.get("params", {})

    params = OFDMParams(
        n_fft=int(p["n_fft"]),
        cp_len=int(p["cp_len"]),
        n_data_subcarriers=(
            None if p.get("n_data_subcarriers", None) is None
            else int(p["n_data_subcarriers"])
        ),
        normalize_ifft=bool(p.get("normalize_ifft", False)),
    )
    return OFDMModulator(name=name, params=params)