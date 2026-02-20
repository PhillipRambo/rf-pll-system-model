from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class RNGManager:
    """
    Simple RNG wrapper to keep reproducibility consistent.
    """
    seed: Optional[int] = None

    def make(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)


def get_rng(seed: Optional[int] = None) -> np.random.Generator:
    """
    Convenience function.
    """
    return np.random.default_rng(seed)
