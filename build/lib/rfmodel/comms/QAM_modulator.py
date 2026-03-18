from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from rfmodel.core.signal import Signal
from rfmodel.core.block import Block


@dataclass
class QAMParams:
    M: int = 64
    gray_map: bool = True
    unit_average_power: bool = True


class QAMModulator(Block):
    """
    Square M-QAM modulator.

    Input
    -----
    s.x : 1D bit array with values {0,1}

    Output
    ------
    Signal with complex QAM symbols in s.x

    Notes
    -----
    - Supports square QAM only: 4, 16, 64, 256, ...
    - For 64-QAM: 6 bits per symbol, split as:
          [b0 b1 b2 | b3 b4 b5]
             I-bits      Q-bits
    - Gray mapping is handled inside the modulator, which is the usual design.
    """

    type_name = "qam_modulator"

    def __init__(self, name: str, params: QAMParams):
        super().__init__(name=name)
        self.params = params

        M = params.M
        k = int(np.log2(M))
        sqrt_M = int(np.sqrt(M))

        if 2**k != M:
            raise ValueError(f"M must be a power of 2, got {M}")
        if sqrt_M * sqrt_M != M:
            raise ValueError(f"M must be square QAM (4,16,64,...), got {M}")
        if k % 2 != 0:
            raise ValueError(f"log2(M) must be even for square QAM, got M={M}")

        self.bits_per_symbol = k
        self.bits_per_axis = k // 2
        self.sqrt_M = sqrt_M

    def process(self, s: Signal) -> Signal:
        bits = np.asarray(s.x)

        if bits.ndim != 1:
            raise ValueError("QAMModulator expects s.x to be a 1D bit array")

        bits = bits.astype(np.uint8)

        if not np.all((bits == 0) | (bits == 1)):
            raise ValueError("Input bits must contain only 0 or 1")

        k = self.bits_per_symbol
        if len(bits) % k != 0:
            raise ValueError(
                f"Number of bits ({len(bits)}) must be a multiple of {k} for {self.params.M}-QAM"
            )

        bit_groups = bits.reshape(-1, k)

        # Split each symbol into I-axis bits and Q-axis bits
        i_bits = bit_groups[:, :self.bits_per_axis]
        q_bits = bit_groups[:, self.bits_per_axis:]

        # Convert bit groups to natural binary indices
        i_idx = self._bits_to_int(i_bits)
        q_idx = self._bits_to_int(q_bits)

        # Apply Gray mapping inside the modulator
        if self.params.gray_map:
            i_idx = self._binary_to_gray(i_idx)
            q_idx = self._binary_to_gray(q_idx)

        # Map indices 0...(sqrt(M)-1) to odd PAM levels
        # Example for 64-QAM:
        # 0..7 -> -7, -5, -3, -1, +1, +3, +5, +7
        i_level = 2 * i_idx.astype(np.int64) - (self.sqrt_M - 1)
        q_level = 2 * q_idx.astype(np.int64) - (self.sqrt_M - 1)

        y = i_level.astype(np.float64) + 1j * q_level.astype(np.float64)

        # Normalize average symbol power to 1
        if self.params.unit_average_power:
            avg_power = (2.0 / 3.0) * (self.params.M - 1)
            y = y / np.sqrt(avg_power)

        meta = dict(s.meta) if s.meta is not None else {}
        meta.update(
            {
                "modulation": f"{self.params.M}-QAM",
                "bits_per_symbol": self.bits_per_symbol,
                "bits_per_axis": self.bits_per_axis,
                "gray_map": self.params.gray_map,
                "unit_average_power": self.params.unit_average_power,
            }
        )

        return s.copy_with(x=y, meta=meta)

    @staticmethod
    def _bits_to_int(b: np.ndarray) -> np.ndarray:
        """
        Convert each row of bits to an integer.

        Example:
            [[1,0,1],
             [0,1,1]]
        ->
            [5, 3]
        """
        weights = (1 << np.arange(b.shape[1] - 1, -1, -1)).astype(np.uint32)
        return (b * weights).sum(axis=1).astype(np.uint32)

    @staticmethod
    def _binary_to_gray(x: np.ndarray) -> np.ndarray:
        """
        Convert natural binary indices to Gray-coded indices.

        Formula:
            gray = binary ^ (binary >> 1)
        """
        x = x.astype(np.uint32)
        return x ^ (x >> 1)
