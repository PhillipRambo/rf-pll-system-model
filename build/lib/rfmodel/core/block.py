from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import numpy as np

from rfmodel.core.signal import Signal


@dataclass
class Block(ABC):
    """
    Base class for all processing blocks in a linear chain.
    """
    name: str
    enabled: bool = True

    def reset(self, seed: Optional[int] = None) -> None:
        """
        Optional: reset internal state (filters, PLL accumulators, RNG, etc.).
        Default: no state.
        """
        _ = seed
        return

    @abstractmethod
    def process(self, s: Signal) -> Signal:
        """
        Transform the input Signal and return a new Signal.
        """
        raise NotImplementedError

    def __call__(self, s: Signal) -> Signal:
        if not self.enabled:
            return s
        return self.process(s)
    

"""
A sub-block that can be used to create a linear model where all blocks are made from same wrapper

Example:
class Gain(Block):
    def __init__(self, gain: float):
        super().__init__(name="gain")
        self.gain = gain

    def process(self, s: Signal) -> Signal:
        return s.copy_with(x=self.gain * s.x)

i.e defining a class and using block as the subclass
"""
