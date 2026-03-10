from __future__ import annotations

import copy
import hashlib
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from abcdef_sim._phys_pipeline import State, hash_ndarray

NDArrayF = npt.NDArray[np.float64]
PHASE_CONTRIBUTIONS_META_KEY: Final[str] = "phase_contributions"


@dataclass(slots=True)
class RayState(State):
    """
    State carrying:
      - rays:   (N,3,1) column vectors following
                ``abcdef_sim.physics.abcdef.conventions``
      - system: (N,3,3) cumulative system matrices for each w
    """

    rays: NDArrayF  # (N,3,1)
    system: NDArrayF  # (N,3,3)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rays = np.asarray(self.rays, dtype=float)
        self.system = np.asarray(self.system, dtype=float)

        if self.rays.ndim == 2 and self.rays.shape[1] == 3:
            # allow (N,3) -> (N,3,1)
            self.rays = self.rays[..., None]

        if self.rays.ndim != 3 or self.rays.shape[-2:] != (3, 1):
            raise ValueError(f"rays must have shape (N,3,1) or (N,3); got {self.rays.shape}")

        if self.system.ndim != 3 or self.system.shape[1:] != (3, 3):
            raise ValueError(f"system must have shape (N,3,3); got {self.system.shape}")

        if self.system.shape[0] != self.rays.shape[0]:
            raise ValueError(
                "system batch dimension must match rays: "
                f"rays.shape={self.rays.shape}, system.shape={self.system.shape}"
            )

    # --- State API ---

    def deepcopy(self) -> RayState:
        return RayState(
            rays=self.rays.copy(),
            system=self.system.copy(),
            meta=copy.deepcopy(self.meta),
        )

    def hashable_repr(self) -> bytes:
        h = hashlib.sha256()
        h.update(hash_ndarray(self.rays))
        h.update(hash_ndarray(self.system))
        phase_contributions = self.meta.get(PHASE_CONTRIBUTIONS_META_KEY)
        if phase_contributions is not None:
            h.update(_hash_phase_contributions_meta(phase_contributions))
        return h.digest()


def _hash_phase_contributions_meta(phase_contributions: Any) -> bytes:
    if not isinstance(phase_contributions, tuple):
        raise TypeError(
            "RayState.meta['phase_contributions'] must be stored as a tuple "
            "for deterministic hashing"
        )

    h = hashlib.sha256()
    for contribution in phase_contributions:
        optic_name = getattr(contribution, "optic_name", None)
        instance_name = getattr(contribution, "instance_name", None)
        backend_id = getattr(contribution, "backend_id", None)
        if not isinstance(optic_name, str) or not isinstance(instance_name, str):
            raise TypeError(
                "Each phase contribution must expose string optic_name and instance_name fields"
            )

        _update_string_hash(h, optic_name)
        _update_string_hash(h, instance_name)
        if backend_id is None:
            h.update(b"\x00")
        elif isinstance(backend_id, str):
            h.update(b"\x01")
            _update_string_hash(h, backend_id)
        else:
            raise TypeError("Phase contribution backend_id must be a string or None")

        _update_array_hash(h, getattr(contribution, "omega", None), name="omega")
        _update_array_hash(
            h,
            getattr(contribution, "delta_omega_rad_per_fs", None),
            name="delta_omega_rad_per_fs",
        )
        _update_array_hash(
            h,
            np.asarray([getattr(contribution, "omega0_rad_per_fs", None)], dtype=float),
            name="omega0_rad_per_fs",
        )
        _update_array_hash(h, getattr(contribution, "phi0_rad", None), name="phi0_rad")
        _update_array_hash(
            h,
            getattr(contribution, "phi_geom_rad", None),
            name="phi_geom_rad",
            allow_none=True,
        )
        _update_array_hash(
            h,
            getattr(contribution, "phi3_transport_like_rad", None),
            name="phi3_transport_like_rad",
            allow_none=True,
        )
        _update_array_hash(
            h,
            getattr(contribution, "phi3_phase_rad", None),
            name="phi3_phase_rad",
            allow_none=True,
        )
        _update_array_hash(h, getattr(contribution, "phi3_rad", None), name="phi3_rad")
        _update_array_hash(
            h,
            getattr(contribution, "path_length_um", None),
            name="path_length_um",
            allow_none=True,
        )
        _update_array_hash(
            h,
            getattr(contribution, "group_delay_fs", None),
            name="group_delay_fs",
            allow_none=True,
        )
        _update_array_hash(
            h,
            getattr(contribution, "filter_amp", None),
            name="filter_amp",
            allow_none=True,
        )
        _update_array_hash(
            h,
            getattr(contribution, "filter_phase_rad", None),
            name="filter_phase_rad",
            allow_none=True,
        )

    return h.digest()


def _update_string_hash(hasher: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    hasher.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
    hasher.update(encoded)


def _update_array_hash(
    hasher: Any,
    value: Any,
    *,
    name: str,
    allow_none: bool = False,
) -> None:
    if value is None:
        if allow_none:
            hasher.update(b"\x00")
            return
        raise TypeError(f"Phase contribution {name} must not be None")

    hasher.update(b"\x01")
    hasher.update(hash_ndarray(np.asarray(value, dtype=float)))
