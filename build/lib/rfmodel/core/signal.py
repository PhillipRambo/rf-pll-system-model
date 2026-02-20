from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Dict, Optional
import numpy as np

@dataclass(frozen=True)
class Signal:
    """
    Container for a complex baseband discrete-time signal plus metadata
    """

    x: np.ndarray       # Stores complex samples
    fs_hz: float        # sample rate [Hz]\

    #metadata 
    fc_hz: Optional[float] = None  # center frequency [Hz] if needed
    meta: Dict[str, Any] = field(default_factory=dict)

    def copy_with(self, **kwargs: Any) -> "Signal":
        """
        Return a modified copy (immutable-style)
        """

        return replace(self, **kwargs)
    
    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])
    
    def ensure_complex(self) -> "Signal":
        if not np.iscomplexobj(self.x):
            return self.copy_with(x=self.x.astype(np.complex128))
        return self
"""
Signal is an immutable container for discrete-time samples and their sampling metadata.
You create it with a NumPy array and a sample rate, then pass it through processing steps that return modified copies instead of changing fields in place.

Key points:
-x holds the samples (real or complex NumPy array).
-fs_hz is the sampling rate.
-fc_hz is optional center frequency metadata.

::: Fields cannot be reassigned after creation (frozen=True).

::: Use copy_with(...) to create a modified version.

::: ensure_complex() converts samples to complex dtype if needed.

::: n_samples returns the number of samples.

Example:

import numpy as np

# Create samples
x = np.cos(2*np.pi*1e3*np.arange(0, 1e-3, 1/100e3))

# Construct signal
sig = Signal(x=x, fs_hz=100e3, meta={"name": "tone"})

# Access derived info
print(sig.n_samples)

# Ensure complex dtype
sig_c = sig.ensure_complex()

# Create a modified copy (scaled signal)
sig_half = sig_c.copy_with(x=0.5 * sig_c.x)
"""