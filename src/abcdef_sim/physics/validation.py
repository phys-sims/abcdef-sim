from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

NDArrayF = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ObservableComparison:
    """Backend-neutral comparison payload for numeric observables."""

    name: str
    observable_label: str
    local: NDArrayF
    reference: NDArrayF
    coordinates: NDArrayF | None = None
    coordinate_label: str | None = None

    def __post_init__(self) -> None:
        local = np.asarray(self.local, dtype=float)
        reference = np.asarray(self.reference, dtype=float)
        if local.shape != reference.shape:
            raise ValueError(
                "local and reference must have the same shape; "
                f"got {local.shape} and {reference.shape}"
            )

        coords: NDArrayF | None = None
        if self.coordinates is not None:
            coords = np.asarray(self.coordinates, dtype=float)
            if coords.shape != local.shape:
                raise ValueError(
                    f"coordinates must match observable shape {local.shape}; got {coords.shape}"
                )

        object.__setattr__(self, "local", local)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "coordinates", coords)

    def residual(self) -> NDArrayF:
        """Return ``local - reference`` for the sampled observable."""

        return np.asarray(self.local - self.reference, dtype=float)

    def max_abs_error(self) -> float:
        """Return the maximum absolute residual."""

        return float(np.max(np.abs(self.residual())))

    def rms_error(self) -> float:
        """Return the RMS residual."""

        residual = self.residual()
        return float(np.sqrt(np.mean(np.square(residual))))
