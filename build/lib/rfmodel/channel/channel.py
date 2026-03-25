# src/rfmodel/channel/channel.py
from __future__ import annotations

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


class ChannelBlock(Block):
    """
    Composite channel block containing an ordered list of channel effects.
    """

    type_name = "channel"

    def __init__(self, name: str, blocks: list[Block]):
        super().__init__(name=name)
        self.blocks = blocks

    def process(self, s: Signal) -> Signal:
        y = s
        for blk in self.blocks:
            y = blk.process(y)
        return y