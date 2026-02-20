from __future__ import annotations

from rfmodel.core.factory import build_block
from rfmodel.core.pipeline import Pipeline

def pipeline_from_config(cfg: dict) -> Pipeline:
    if "pipeline" not in cfg:
        raise KeyError("Config missing top-level key: 'pipeline'")
    pipe = Pipeline()
    for block_cfg in cfg["pipeline"]:
        pipe.add(build_block(block_cfg))
    return pipe


