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
        Resets all blocks.
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
========================
Pipeline usage guide
========================

Overview
--------
Pipeline is a linear chain of Blocks. A Block transforms a Signal and returns a new Signal.

Conceptually:
    s_out = blockN(... block2(block1(s_in)) ...)

Each Block must have:
  - a unique name (string)
  - be callable: block(signal) -> signal
  - an 'enabled' flag (bool) that the Block __call__ uses to bypass processing when disabled
  - an optional reset(seed=...) method for reinitializing internal state (RNG, filter state, etc.)

The Pipeline supports:
  - adding blocks (append / insert before / insert after)
  - removing blocks
  - replacing blocks
  - enabling/disabling blocks
  - resetting all blocks
  - running the chain and optionally capturing intermediate ("tapped") signals


Quick start
----------
1) Create a pipeline and add blocks in order:

    pipe = Pipeline()
    pipe.add(SourceBlock(name="src", params=...))
    pipe.add(LNABlock(name="lna", params=...))
    pipe.add(FilterBlock(name="chan_filt", params=...))
    pipe.add(ADCBlock(name="adc", params=...))

The internal order is exactly the order blocks are added (unless you insert before/after).


Adding blocks
-------------
Append (default):

    pipe.add(my_block)

Insert before a named block:

    pipe.add(my_block, before="lna")   # inserted immediately before the block named "lna"

Insert after a named block:

    pipe.add(my_block, after="lna")    # inserted immediately after the block named "lna"

Notes:
  - You must specify only one of 'before' or 'after'.
  - The target name must exist or a KeyError is raised.
  - In practice, block names should be unique to avoid ambiguity.


Removing blocks
---------------
Remove a block by name:

    pipe.remove("chan_filt")

If the name is not found, KeyError is raised.


Replacing blocks
----------------
Replace an existing block (keeps its position in the chain):

    pipe.replace("lna", LNABlock(name="lna", params=new_params))

Note:
  - The replacement block should usually keep the same name if downstream code refers to that name (e.g., taps).


Enable / disable blocks
-----------------------
Disable a block (Block should act as a bypass when disabled):

    pipe.enable("lna", enabled=False)

Re-enable:

    pipe.enable("lna", enabled=True)

Important:
  - This relies on Block.__call__ implementing the bypass behavior when enabled == False.


Resetting blocks
----------------
Reset all blocks:

    pipe.reset()

Reset all blocks with a deterministic seed:

    pipe.reset(seed=1234)

Typical uses:
  - reset RNG state for noise blocks
  - clear filter states / memory
  - restart any internal block state


Running the pipeline
--------------------
Run a signal through the full chain:

    s_out, tapped = pipe.run(s_in)

By default, nothing is tapped:

    tapped == {}

Tap intermediate signals at specific block outputs:

    s_out, tapped = pipe.run(s_in, taps=["lna", "adc"])

Then:
    tapped["lna"] is the Signal immediately after the block named "lna"
    tapped["adc"] is the Signal immediately after the block named "adc"

Visual model:
    s_in -> [src] -> [lna] -> [chan_filt] -> [adc] -> s_out
                       ^                      ^
                       |                      |
                  tapped["lna"]          tapped["adc"]

Notes:
  - Taps capture the Signal object returned by each block at that point.
  - Best practice is that blocks return a new Signal (immutable style). If blocks mutate in place,
    taps may be affected by later processing.


Common patterns
---------------
A) Debugging: capture signals to inspect power/spectrum between blocks:

    s_out, t = pipe.run(s_in, taps=["src", "lna", "chan_filt", "adc"])
    plot_psd(t["lna"].x, fs=t["lna"].fs_hz)
    plot_psd(t["chan_filt"].x, fs=t["chan_filt"].fs_hz)

B) A/B comparison: disable one block and compare output:

    pipe.enable("chan_filt", False)
    out_no_filt, _ = pipe.run(s_in)

    pipe.enable("chan_filt", True)
    out_with_filt, _ = pipe.run(s_in)

C) Swap models: replace a block with a different implementation:

    pipe.replace("lna", LNABlock(name="lna", params=lna_params_v2))


Error behavior
--------------
- add(before=..., after=...) with both set -> ValueError
- referencing a missing block name -> KeyError


Minimal example template
------------------------
    pipe = Pipeline()
    pipe.add(BlockA(name="a", params=...))
    pipe.add(BlockB(name="b", params=...))
    pipe.add(BlockC(name="c", params=...))

    pipe.reset(seed=0)

    s_out, taps = pipe.run(s_in, taps=["b"])
    s_b = taps["b"]


End
---
This guide assumes your Block base class:
  - defines .name and .enabled
  - implements __call__(Signal) -> Signal
  - provides reset(seed=...) or a no-op default
"""