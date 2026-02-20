from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from rfmodel.core.block import Block
from rfmodel.core.signal import Signal


@dataclass
class Pipeline:
    """
    Linear ordered chain of blocks.
    """
    blocks: List[Block] = field(default_factory=list)

    def add(self, block: Block, *, before: Optional[str] = None, after: Optional[str] = None) -> None:
        if before and after:
            raise ValueError("Specify only one of before or after.")
        if before is None and after is None:
            self.blocks.append(block)
            return

        idx = self._index_of(before or after)
        if before is not None:
            self.blocks.insert(idx, block)
        else:
            self.blocks.insert(idx + 1, block)

    def remove(self, name: str) -> None:
        idx = self._index_of(name)
        self.blocks.pop(idx)

    def replace(self, name: str, new_block: Block) -> None:
        idx = self._index_of(name)
        self.blocks[idx] = new_block

    def enable(self, name: str, enabled: bool = True) -> None:
        self.blocks[self._index_of(name)].enabled = enabled

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Resets all blocks. If seed is provided, you can choose a scheme later.
        For now, just pass the same seed to all blocks.
        """
        for b in self.blocks:
            b.reset(seed=seed)

    def run(self, s: Signal, *, taps: Optional[Sequence[str]] = None) -> tuple[Signal, Dict[str, Signal]]:
        """
        Run the pipeline. Optionally capture intermediate signals at named blocks.

        Returns:
          (final_signal, tapped_signals)
        """
        taps_set = set(taps) if taps else set()
        captured: Dict[str, Signal] = {}

        cur = s
        for b in self.blocks:
            cur = b(cur)
            if b.name in taps_set:
                captured[b.name] = cur
        return cur, captured

    def _index_of(self, name: str) -> int:
        for i, b in enumerate(self.blocks):
            if b.name == name:
                return i
        raise KeyError(f"Block '{name}' not found in pipeline.")

"""
The linearized Pipeline, that manages the model structure.
"""