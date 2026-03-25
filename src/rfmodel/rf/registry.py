from __future__ import annotations
import numpy as np

from rfmodel.core.factory import register_block
from rfmodel.rf.LNA import LNABlock, LNAParams
from rfmodel.rf.PA import PABlock, PAParams


@register_block("lna")
def _build_lna(cfg: dict) -> LNABlock:
    name = cfg["name"]
    seed = cfg.get("seed", None)
    p = cfg.get("params", {})

    params = LNAParams(
        gain_db=float(p["gain_db"]),
        nf_db=float(p["nf_db"]),
        IP3_dbm=float(p["IP3_dbm"]),
        temp_k=float(p.get("temp_k", 290.0)),
    )
    return LNABlock(name=name, params=params, seed=seed)



@register_block("pa")
def _build_pa(cfg: dict) -> PABlock:
    name = cfg["name"]
    p = cfg.get("params", {})

    params = PAParams(
        gain_db=float(p["gain_db"]),
        p1db_out_dbm=float(p["p1db_out_dbm"]),
        smoothness_p=float(p.get("smoothness_p", 2.0)),
    )
    return PABlock(name=name, params=params)