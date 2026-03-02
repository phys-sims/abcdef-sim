from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np

from abcdef_sim.optics.base import ArrayLike, NDArrayF, Optic, RefractiveIndexFn
from abcdef_sim.utils.optics_builder import get_abcdef_matrices


def _coerce_length(value: float | int | str | bytes | bytearray) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"Length must be numeric, got {type(value).__name__}") from exc


@dataclass(slots=True, init=False)
class FreeSpace(Optic):
    """Free-space propagation optic."""

    def __init__(
        self,
        *args: object,
        length: float | None = None,
        name: str = "FreeSpace",
        instance_name: str = "inst0",
        _length: float | None = None,
        _n_fn: RefractiveIndexFn | None = None,
    ) -> None:
        """Initialize free space with backward-compatible constructor behavior.

        Supported forms:
        - New: ``FreeSpace(length=..., name=..., instance_name=..., _n_fn=...)``
        - Legacy keyword: ``FreeSpace(_length=..., ...)``
        - Legacy positional (old dataclass order):
          ``FreeSpace(name, instance_name, _length, _n_fn)``
        """
        if len(args) > 4:
            raise TypeError(f"FreeSpace expected at most 4 positional args, got {len(args)}")

        if args:
            if len(args) >= 1:
                name = str(args[0])
            if len(args) >= 2:
                instance_name = str(args[1])
            if len(args) >= 3:
                _length = _coerce_length(cast(float | int | str | bytes | bytearray, args[2]))
            if len(args) == 4:
                _n_fn = cast(RefractiveIndexFn | None, args[3])

        if length is not None and _length is not None:
            raise TypeError("Provide only one of 'length' or '_length', not both")

        if length is not None:
            resolved_length = float(length)
        elif _length is not None:
            resolved_length = float(_length)
        else:
            resolved_length = 0.0

        self.name = name
        self.instance_name = instance_name
        self._length = resolved_length
        self._n_fn = _n_fn

    def matrix(self, omega: ArrayLike) -> NDArrayF:
        omega_arr = np.asarray(omega, dtype=np.float64)
        return get_abcdef_matrices(a=1.0, b=self.length, c=0.0, d=1.0, omega=omega_arr)
