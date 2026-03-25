from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class PRBSParams:
    order: int                  # supported: 7 or 15
    n_bits: int
    seed: int = 1
    output_dtype: np.dtype = np.uint8


class PRBSBitSource(Block):
    type_name = "prbs_bit_source"

    # Polynomial taps:
    # PRBS7  -> x^7  + x^6  + 1
    # PRBS15 -> x^15 + x^14 + 1
    _TAPS = {
        7: (7, 6),
        15: (15, 14),
    }

    def __init__(self, name: str, params: PRBSParams):
        super().__init__(name=name)
        self.params = params

        if params.order not in self._TAPS:
            raise ValueError("Supported PRBS orders are 7 and 15")
        if params.n_bits <= 0:
            raise ValueError("n_bits must be > 0")

        max_seed = (1 << params.order) - 1
        if not (1 <= params.seed <= max_seed):
            raise ValueError(
                f"seed must be in [1, {max_seed}] for PRBS{params.order}"
            )

        self.order = params.order
        self.n_bits = params.n_bits
        self.seed = params.seed
        self.output_dtype = params.output_dtype
        self.taps = self._TAPS[self.order]
        self.mask = (1 << self.order) - 1

    def _generate_bits(self) -> np.ndarray:
        state = self.seed
        t1, t2 = self.taps
        y = np.empty(self.n_bits, dtype=self.output_dtype)

        for i in range(self.n_bits):
            # Output bit = MSB of current state
            y[i] = (state >> (self.order - 1)) & 1

            # Feedback from polynomial taps
            b1 = (state >> (t1 - 1)) & 1
            b2 = (state >> (t2 - 1)) & 1
            feedback = b1 ^ b2

            # Shift left, insert feedback into LSB
            state = ((state << 1) & self.mask) | feedback

        return y

    def process(self, s: Signal) -> Signal:
        y = self._generate_bits()

        meta = dict(s.meta) if s.meta is not None else {}
        meta.update({
            "source": f"PRBS{self.order}",
            "prbs_order": self.order,
            "n_bits": self.n_bits,
            "seed": self.seed,
        })

        return s.copy_with(x=y, meta=meta)