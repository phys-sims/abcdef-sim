from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from abcdef_sim.physics.abcd.ray import Ray

NDArrayF = npt.NDArray[np.float64]


def _import_raytracing() -> Any:
    try:
        import raytracing as rt
    except ImportError as exc:
        raise ImportError(
            "raytracing is required for reference validation helpers. "
            "Install it with `pip install -e '.[validation]'`."
        ) from exc
    return rt


def to_raytracing_matrix(M: NDArrayF) -> Any:
    """Convert a local 2x2 ABCD matrix into a raytracing.Matrix instance."""

    rt = _import_raytracing()
    M_arr = np.asarray(M, dtype=float)
    if M_arr.shape != (2, 2):
        raise ValueError(f"M must have shape (2, 2); got {M_arr.shape}")
    return rt.Matrix(A=M_arr[0, 0], B=M_arr[0, 1], C=M_arr[1, 0], D=M_arr[1, 1])


def from_raytracing_matrix(rt_matrix: Any) -> NDArrayF:
    """Convert a raytracing.Matrix-like object into a local numpy 2x2 matrix."""

    return np.array(
        [
            [float(rt_matrix.A), float(rt_matrix.B)],
            [float(rt_matrix.C), float(rt_matrix.D)],
        ],
        dtype=float,
    )


def abcd_from_raytracing(obj: Any) -> NDArrayF:
    """Extract a local 2x2 ABCD matrix from a raytracing matrix-like object."""

    return from_raytracing_matrix(obj)


def raytracing_space(d: float, n: float = 1.0) -> Any:
    """Build a raytracing native space element for oracle-mode validation."""

    rt = _import_raytracing()
    return rt.Space(d=d, n=n)


def raytracing_thin_lens(f: float) -> Any:
    """Build a raytracing native thin lens for oracle-mode validation."""

    rt = _import_raytracing()
    return rt.Lens(f=f)


def raytracing_interface(n1: float, n2: float, R: float | None) -> Any:
    """Build a raytracing dielectric interface for oracle-mode validation."""

    rt = _import_raytracing()
    return rt.DielectricInterface(n1=n1, n2=n2, R=np.inf if R is None else R)


def raytracing_thick_lens(
    n: float,
    R1: float | None,
    R2: float | None,
    thickness: float,
    n_in: float = 1.0,
    n_out: float = 1.0,
) -> Any:
    """Build a raytracing native thick lens element for oracle-mode validation.

    For an air-to-air lens with finite radii on both sides this uses
    ``rt.ThickLens`` directly. Otherwise the lens is assembled from native
    interfaces and a propagation space via matrix multiplication.
    """

    rt = _import_raytracing()
    if n_in == 1.0 and n_out == 1.0 and R1 is not None and R2 is not None:
        return rt.ThickLens(n=n, R1=R1, R2=R2, thickness=thickness)

    front = raytracing_interface(n1=n_in, n2=n, R=R1)
    middle = raytracing_space(d=thickness, n=n)
    back = raytracing_interface(n1=n, n2=n_out, R=R2)
    return back * middle * front


def raytracing_compose(*elements: Any) -> Any:
    """Compose raytracing-native elements in optical traversal order."""

    rt = _import_raytracing()
    total = rt.Matrix(A=1.0, B=0.0, C=0.0, D=1.0)
    for element in elements:
        total = element * total
    return total


def raytracing_gaussian_beam(q: complex, wavelength: float) -> Any:
    """Build a raytracing GaussianBeam with an explicit q-parameter."""

    rt = _import_raytracing()
    return rt.GaussianBeam(q=q, wavelength=wavelength)


def propagate_gaussian_beam_raytracing(
    q: complex,
    wavelength: float,
    elements: Sequence[Any],
) -> Any:
    """Propagate a Gaussian beam through native raytracing elements."""

    beam = raytracing_gaussian_beam(q=q, wavelength=wavelength)
    return raytracing_compose(*elements) * beam


def sample_gaussian_beam_radii_raytracing(beam: Any, z_samples: npt.ArrayLike) -> NDArrayF:
    """Sample Gaussian beam radii after extra free-space propagation distances."""

    rt = _import_raytracing()
    z_arr = np.asarray(z_samples, dtype=float)
    radii = np.empty_like(z_arr, dtype=float)
    for idx in np.ndindex(z_arr.shape):
        radii[idx] = float((rt.Space(d=float(z_arr[idx])) * beam).w)
    return radii


def trace_ray_raytracing(ray: Ray, elements: Sequence[NDArrayF]) -> Ray:
    """Trace a ray using raytracing as a local matrix engine (mirror mode).

    The ``elements`` sequence must contain local 2x2 matrices in traversal order.
    """

    rt = _import_raytracing()
    rt_ray = rt.Ray(y=ray.y, theta=ray.theta)
    total = raytracing_compose(*(to_raytracing_matrix(element) for element in elements))

    out = total * rt_ray
    return Ray(y=float(out.y), theta=float(out.theta))
