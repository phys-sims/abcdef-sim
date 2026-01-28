import numpy as np
import numpy.typing as npt
from typing import Callable, Optional

ArrayLike = npt.ArrayLike
NDArrayF = npt.NDArray[np.float64]


def get_abcdef_matrices(
    a: float,
    b: float,
    c: float,
    d: float,
    omega: ArrayLike,
    e_of_omega: Optional[Callable[[NDArrayF], NDArrayF]] = None,
    f_of_omega: Optional[Callable[[NDArrayF], NDArrayF]] = None,
) -> NDArrayF:
    """
    Build an array of 3x3 ABCDEF matrices for a frequency grid.

    For each ω_n:

        [ a   b   E(ω_n) ]
        [ c   d   F(ω_n) ]
        [ 0   0     1    ]

    If e_of_omega or f_of_omega is None, the corresponding E or F
    is taken as 0 for all ω.
    """
    w = np.asarray(omega, dtype=np.float64)

    scalar = (w.ndim == 0)
    if scalar:
        w = w[None]

    E = np.zeros_like(w) if e_of_omega is None else np.asarray(e_of_omega(w), dtype=np.float64)
    F = np.zeros_like(w) if f_of_omega is None else np.asarray(f_of_omega(w), dtype=np.float64)
    if E.shape != w.shape or F.shape != w.shape:
        raise ValueError(f"E,F must match omega shape {w.shape}, got {E.shape}, {F.shape}")

    M = np.zeros((w.size, 3, 3), dtype=np.float64)
    M[:, 0, 0] = a
    M[:, 0, 1] = b
    M[:, 1, 0] = c
    M[:, 1, 1] = d
    M[:, 2, 2] = 1.0
    M[:, 0, 2] = E
    M[:, 1, 2] = F

    return M[0] if scalar else M