from __future__ import annotations

from typing import Any, Callable, Dict

from rfmodel.core.block import Block

# type -> constructor(cfg_dict) -> Block
_REGISTRY: Dict[str, Callable[[dict], Block]] = {}


def register_block(block_type: str):
    """
    Decorator to register a block builder.
    The builder takes the YAML dict for a block and returns a Block instance.
    """
    def _wrap(fn: Callable[[dict], Block]):
        _REGISTRY[block_type] = fn
        return fn
    return _wrap


def build_block(block_cfg: dict) -> Block:
    block_type = block_cfg.get("type")
    if not block_type:
        raise ValueError("Block config missing 'type'")
    if block_type not in _REGISTRY:
        raise KeyError(f"Unknown block type '{block_type}'. Registered: {list(_REGISTRY.keys())}")
    return _REGISTRY[block_type](block_cfg)
