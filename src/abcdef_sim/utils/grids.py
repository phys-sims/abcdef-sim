from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True)
class LinspaceGrid:
    """
    Represents omega = linspace(w0-span, w0+span, N)
    """
    w0: float
    span: float
    N: int

    def omega(self) -> NDArrayF:
        return np.linspace(self.w0 - self.span, self.w0 + self.span, self.N, dtype=np.float64)

    @property
    def w_min(self) -> float:
        return float(self.w0 - self.span)

    @property
    def w_max(self) -> float:
        return float(self.w0 + self.span)

    @property
    def dw(self) -> float:
        if self.N <= 1:
            return 0.0
        return float((self.w_max - self.w_min) / (self.N - 1))

# VERIFY
def infer_linspace_grid(w: NDArrayF, atol: float = 1e-12) -> Optional[LinspaceGrid]:
    """
    Attempt to infer LinspaceGrid from an omega array.
    Returns None if not evenly spaced.
    """
    w = np.asarray(w, dtype=np.float64).reshape(-1)
    N = w.size
    if N == 0:
        return None
    if N == 1:
        return LinspaceGrid(w0=float(w[0]), span=0.0, N=1)

    d = np.diff(w)
    if not np.allclose(d, d[0], atol=atol, rtol=0.0):
        return None

    w_min = float(w[0])
    w_max = float(w[-1])
    w0 = 0.5 * (w_min + w_max)
    span = 0.5 * (w_max - w_min)
    return LinspaceGrid(w0=w0, span=span, N=N)
