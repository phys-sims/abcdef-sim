"""Pure Martinez phase bookkeeping helpers.

Equation references are from the private Martinez paper:
``Matrix Formalism for Pulse Compressors`` (IEEE JQE, 1988), eqs. 24-30.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcdef.conventions import (
    extract_A,
    extract_B,
    validate_matrix_shape,
    validate_ray_shape,
)

NDArrayF = npt.NDArray[np.float64]
NDArrayC = npt.NDArray[np.complex128]

__all__ = [
    "combine_phi_total_rad",
    "phi0_rad_i",
    "phi1_rad",
    "phi2_rad",
    "phi3_rad_i",
    "phi4_rad",
]


def phi0_rad_i(k: object, length: float, n: object) -> NDArrayF:
    """Return the per-element plane-wave phase from Martinez eq. 24."""

    k_arr = _as_real_vector("k", k)
    n_arr = _as_real_vector("n", n)
    _require_matching_size(("k", k_arr), ("n", n_arr))
    return k_arr * float(length) * n_arr


def phi3_rad_i(k: object, F_i: object, x_after: object) -> NDArrayF:
    """Return the per-element angular-dispersion phase from Martinez eq. 27."""

    k_arr = _as_real_vector("k", k)
    f_arr = _as_real_vector("F_i", F_i)
    x_arr = _as_real_vector("x_after", x_after)
    _require_matching_size(("k", k_arr), ("F_i", f_arr), ("x_after", x_arr))
    return 0.5 * k_arr * f_arr * x_arr


def phi2_rad(k: object, ray_in: object, ray_out: object) -> NDArrayF:
    """Return the input/output ray-centering phase from Martinez eq. 26."""

    k_arr = _as_real_vector("k", k)
    rays_in = validate_ray_shape(ray_in)
    rays_out = validate_ray_shape(ray_out)

    batch_size = k_arr.size
    if rays_in.shape[0] != batch_size or rays_out.shape[0] != batch_size:
        raise ValueError(
            "k, ray_in, and ray_out must have matching batch sizes: "
            f"k.shape={k_arr.shape}, ray_in.shape={rays_in.shape}, "
            f"ray_out.shape={rays_out.shape}"
        )

    x_in = rays_in[:, 0, 0]
    x_prime_in = rays_in[:, 1, 0]
    x_out = rays_out[:, 0, 0]
    x_prime_out = rays_out[:, 1, 0]

    return 0.5 * k_arr * (x_in * x_prime_in - x_out * x_prime_out)


def phi1_rad(abcdef: object, q_in: object, w_in: object, w_out: object) -> NDArrayF:
    """Return the Gaussian-beam correction phase from Martinez eq. 25."""

    matrices = validate_matrix_shape(abcdef)
    if matrices.ndim == 2:
        matrices = matrices[None, ...]

    q_arr = _as_complex_vector("q_in", q_in)
    w_in_arr = _as_real_vector("w_in", w_in)
    w_out_arr = _as_real_vector("w_out", w_out)

    batch_size = matrices.shape[0]
    if q_arr.size != batch_size or w_in_arr.size != batch_size or w_out_arr.size != batch_size:
        raise ValueError(
            "abcdef, q_in, w_in, and w_out must have matching batch sizes: "
            f"abcdef.shape={matrices.shape}, q_in.shape={q_arr.shape}, "
            f"w_in.shape={w_in_arr.shape}, w_out.shape={w_out_arr.shape}"
        )

    if np.any(w_in_arr <= 0.0) or np.any(w_out_arr <= 0.0):
        raise ValueError("w_in and w_out must be strictly positive")

    z = extract_A(matrices).astype(np.complex128) + extract_B(matrices) / q_arr
    lhs = 1.0 / np.sqrt(z)
    amplitude_factor = np.sqrt(w_in_arr / w_out_arr)
    normalized_phase = lhs / amplitude_factor
    if not np.allclose(np.abs(normalized_phase), 1.0, rtol=1e-9, atol=1e-12):
        raise ValueError(
            "sqrt(w_in / w_out) must match |1 / sqrt(A + B / q_in)| from Martinez eq. 25"
        )

    return -np.angle(normalized_phase)


def phi4_rad(k: object, x: object, x_out: object, q_out: object) -> NDArrayF:
    """Return the spatial phase term from Martinez eq. 30."""

    k_arr = _as_real_vector("k", k)
    x_arr = _broadcast_real_vector("x", x, size=k_arr.size)
    x_out_arr = _as_real_vector("x_out", x_out)
    q_out_arr = _as_complex_vector("q_out", q_out)
    _require_matching_size(("k", k_arr), ("x_out", x_out_arr), ("q_out", q_out_arr))

    return np.real(k_arr * (x_arr - x_out_arr) ** 2 / (2.0 * q_out_arr))


def combine_phi_total_rad(*terms: object | None) -> NDArrayF:
    """Sum any provided phase terms into a total phase array."""

    arrays = [
        _as_real_vector(f"terms[{i}]", term) for i, term in enumerate(terms) if term is not None
    ]
    if not arrays:
        raise ValueError("combine_phi_total_rad requires at least one non-None term")

    named_arrays = [(f"terms[{i}]", arr) for i, arr in enumerate(arrays)]
    _require_matching_size(*named_arrays)

    total = np.zeros_like(arrays[0], dtype=np.float64)
    for arr in arrays:
        total = total + arr
    return total


def _as_real_vector(name: str, value: object) -> NDArrayF:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be coercible to shape (N,); got {arr.shape}")
    return arr


def _as_complex_vector(name: str, value: object) -> NDArrayC:
    arr = np.asarray(value, dtype=np.complex128).reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be coercible to shape (N,); got {arr.shape}")
    return arr


def _broadcast_real_vector(name: str, value: object, *, size: int) -> NDArrayF:
    arr = _as_real_vector(name, value)
    if arr.size == size:
        return arr
    if arr.size == 1:
        return np.full(size, arr.item(), dtype=np.float64)
    raise ValueError(f"{name} must have shape ({size},); got {arr.shape}")


def _require_matching_size(*named_arrays: tuple[str, npt.NDArray[np.generic]]) -> None:
    expected_size = named_arrays[0][1].size
    for name, arr in named_arrays[1:]:
        if arr.size != expected_size:
            raise ValueError(f"{name} must have shape ({expected_size},); got {arr.shape}")
