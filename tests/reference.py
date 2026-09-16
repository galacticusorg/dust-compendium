"""Independent references for the geometry tests.

Everything here is computed by numerical quadrature straight from the density
definitions, so that it shares no code with what it is checking. The point is
that a closed form and its reference can only agree by being right.
"""

import numpy as np
from scipy.integrate import dblquad, quad

HERNQUIST = lambda x: 1.0 / (x * (1.0 + x) ** 3)  # noqa: E731
JAFFE = lambda x: 1.0 / (x**2 * (1.0 + x) ** 2)  # noqa: E731


def spherical_cell_mass(shape, scale_radial, radius_lower, radius_upper, height_lower, height_upper):
    r"""Mass of an annular cell for a spherical profile, per unit normalization."""
    # Split at the midplane when the cell straddles it: the integrand has a
    # cusp there, and integrating across it costs several digits.
    if height_lower < 0.0 < height_upper:
        return spherical_cell_mass(
            shape, scale_radial, radius_lower, radius_upper, height_lower, 0.0
        ) + spherical_cell_mass(shape, scale_radial, radius_lower, radius_upper, 0.0, height_upper)
    value, _ = dblquad(
        lambda radius, height: radius * shape(np.hypot(radius, height) / scale_radial),
        height_lower,
        height_upper,
        lambda _: radius_lower,
        lambda _: radius_upper,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )
    return 2.0 * np.pi * value


def separable_cell_mass(radial, vertical, radius_lower, radius_upper, height_lower, height_upper):
    """Mass of an annular cell for a profile separable in radius and height."""
    radial_part, _ = quad(
        lambda radius: radius * radial(radius),
        radius_lower,
        radius_upper,
        epsabs=1.0e-12,
        epsrel=1.0e-12,
    )
    vertical_part, _ = quad(vertical, height_lower, height_upper, epsabs=1.0e-12, epsrel=1.0e-14)
    return 2.0 * np.pi * radial_part * vertical_part


def column_along_ray(density, radius):
    r"""The column :math:`\int_{-\infty}^{\infty} \hat\rho(R, z)\,\mathrm{d}z` at a radius."""
    value, _ = quad(
        lambda height: density(radius, height),
        -np.inf,
        np.inf,
        epsabs=1.0e-13,
        epsrel=1.0e-13,
    )
    return value
