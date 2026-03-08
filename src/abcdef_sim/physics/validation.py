from __future__ import annotations

from collections.abc import Sequence
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


@dataclass(frozen=True, slots=True)
class BenchmarkComparison:
    """Comparable wall-clock timings for one validation scenario."""

    scenario_name: str
    wavelength_count: int
    local_seconds: float
    reference_seconds: float
    local_label: str = "abcdef-sim"
    reference_label: str = "raytracing"

    def __post_init__(self) -> None:
        local_seconds = float(self.local_seconds)
        reference_seconds = float(self.reference_seconds)
        if local_seconds <= 0.0:
            raise ValueError("local_seconds must be > 0")
        if reference_seconds <= 0.0:
            raise ValueError("reference_seconds must be > 0")

        object.__setattr__(self, "local_seconds", local_seconds)
        object.__setattr__(self, "reference_seconds", reference_seconds)

    def speedup_factor(self) -> float:
        """Return reference/backend speedup using ``reference / local``."""

        return float(self.reference_seconds / self.local_seconds)


def format_benchmark_table(comparisons: Sequence[BenchmarkComparison]) -> str:
    """Render benchmark comparisons as a Markdown table."""

    rows = list(comparisons)
    if not rows:
        raise ValueError("comparisons must be non-empty")

    header = [
        "| Scenario | Wavelengths | abcdef-sim (ms) | raytracing (ms) | Speedup |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    body = [
        (
            f"| {row.scenario_name} | {row.wavelength_count} | "
            f"{row.local_seconds * 1e3:.3f} | {row.reference_seconds * 1e3:.3f} | "
            f"{row.speedup_factor():.2f}x |"
        )
        for row in rows
    ]
    return "\n".join([*header, *body])
