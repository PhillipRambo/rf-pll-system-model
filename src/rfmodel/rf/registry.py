from __future__ import annotations

from rfmodel.core.factory import register_block
from rfmodel.rf.LNA import LNABlock, LNAParams


@register_block("lna")
def _build_lna(cfg: dict) -> LNABlock:
    name = cfg["name"]
    seed = cfg.get("seed", None)
    p = cfg.get("params", {})
    params = LNAParams(
        gain_db=float(p["gain_db"]),
        nf_db=float(p["nf_db"]),
        temp_k=float(p.get("temp_k", 290.0)),
    )
    return LNABlock(name=name, params=params, seed=seed)
