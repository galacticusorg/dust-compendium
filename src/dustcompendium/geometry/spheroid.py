r"""Spheroid profiles: spherical Hernquist and Jaffe distributions.

Both are spherical, written in terms of :math:`x = r/r_\mathrm{s}` with
:math:`r = \sqrt{R^2 + z^2}`:

.. math::

    \hat\rho_\mathrm{Hernquist}(x) = \frac{1}{x (1 + x)^3},
    \qquad
    \hat\rho_\mathrm{Jaffe}(x) = \frac{1}{x^2 (1 + x)^2}.

Both diverge at the centre, so the optical depth through the centre is infinite
and cannot be used to normalize them. Optical depth is instead defined along a
ray at the scale radius -- see :attr:`Profile.optical_depth_radius`. This
convention is inherited from the original tabulations and is the one the
published datasets use; it is *not* a central optical depth, and a spheroid
optical depth must be interpreted accordingly.

Cell masses come from closed-form antiderivatives of :math:`\hat\rho R` in
cylindrical coordinates. These carry removable singularities at :math:`R = 0`
and at :math:`R = r_\mathrm{s}`, handled by separate expressions, and are
evaluated through a complex continuation of :math:`\sqrt{R^2 - r_\mathrm{s}^2}`
so that a single expression covers :math:`R` inside and outside the scale
radius. The imaginary parts cancel; only the real part is kept.
"""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import quad

from .base import Profile

__all__ = ["HernquistSpheroid", "JaffeSpheroid", "Spheroid"]

# Distances, in units of the scale radius, within which the closed-form
# antiderivative is too ill-conditioned to use, and a cell is integrated
# numerically instead. Each was chosen by measuring the closed form against
# quadrature as a cell corner approaches the locus; see
# tests/test_geometry_spheroid.py, which re-measures them.
#
# The sphere r = r_s is the important one. The generic Hernquist expression
# divides by x^2 + u^2 - 1, and the generic Jaffe expression is likewise
# singular there. Both are removable -- the numerators vanish too -- but the
# original implementation special-cased only R = 0 and R = r_s, leaving the rest
# of the sphere to evaluate as 0/0. A cell corner landing exactly on it, such as
# (0.6, 0.8) r_s, returns NaN.
# Measured worst-case relative error of the closed form at the edge of each
# band, against quadrature: sphere 1.5e-11, scale radius 1.1e-11, axis 1.6e-14.
# Just inside them the closed form degrades quickly -- Hernquist reaches 5.5e-8
# at 1e-5 from the scale radius and 1.9e+03 at 1e-12.
_SPHERE_TOLERANCE = 1.0e-4
_AXIS_TOLERANCE = 1.0e-7
_SCALE_TOLERANCE = 1.0e-3


class Spheroid(Profile):
    r"""Common machinery for the spherical profiles.

    Parameters
    ----------
    scale_radial
        The scale radius :math:`r_\mathrm{s}`, positive.
    """

    #: The column of the shape function along a ray at the scale radius, in
    #: units of the scale radius. Set by each subclass.
    _COLUMN_AT_SCALE_RADIUS: float

    def __init__(self, scale_radial: float) -> None:
        if scale_radial <= 0.0:
            raise ValueError(f"scale radius must be positive, got {scale_radial}")
        self.scale_radial = float(scale_radial)

    def _shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """The shape function in units of the scale radius."""
        raise NotImplementedError

    def _enclosed_annulus(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""The exact inner integral :math:`P(x) = \int \hat\rho(x) x \,\mathrm{d}x`.

        Integrating over radius at fixed height is exact for a spherical
        profile: substituting :math:`r^2 = R^2 + z^2` turns
        :math:`\int \hat\rho R \,\mathrm{d}R` into :math:`P` evaluated at the
        two radii. Only the remaining integral over height needs quadrature,
        which is what the fallback in :meth:`cell_mass` uses.
        """
        raise NotImplementedError

    def _antiderivative(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        r"""Antiderivative of :math:`\hat\rho R` in both arguments, in scale-radius units.

        Returns :math:`F(x, u)` such that
        :math:`\int\int \hat\rho R \,\mathrm{d}R\,\mathrm{d}z = F`, with
        :math:`x = R/r_\mathrm{s}` and :math:`u = z/r_\mathrm{s}`.
        """
        raise NotImplementedError

    def density(self, radius: ArrayLike, height: ArrayLike) -> NDArray[np.float64]:
        r = np.asarray(radius, dtype=float)
        z = np.asarray(height, dtype=float)
        x = np.sqrt(r * r + z * z) / self.scale_radial
        with np.errstate(divide="ignore", invalid="ignore"):
            return self._shape(x)

    def cell_mass(
        self,
        radius_lower: ArrayLike,
        radius_upper: ArrayLike,
        height_lower: ArrayLike,
        height_upper: ArrayLike,
    ) -> NDArray[np.float64]:
        scale = self.scale_radial
        x_lower = np.asarray(radius_lower, dtype=float) / scale
        x_upper = np.asarray(radius_upper, dtype=float) / scale
        u_lower = np.asarray(height_lower, dtype=float) / scale
        u_upper = np.asarray(height_upper, dtype=float) / scale
        x_lower, x_upper, u_lower, u_upper = np.broadcast_arrays(x_lower, x_upper, u_lower, u_upper)
        # The plain two-dimensional finite difference of the antiderivative. The
        # original implementation combined the corners through nested absolute
        # values, which silently returns zero for a cell symmetric about the
        # midplane; see the regression test.
        with np.errstate(divide="ignore", invalid="ignore"):
            difference = (
                self._antiderivative(x_upper, u_upper)
                - self._antiderivative(x_lower, u_upper)
                - self._antiderivative(x_upper, u_lower)
                + self._antiderivative(x_lower, u_lower)
            )
        mass = 2.0 * np.pi * scale**3 * difference

        # Fall back to quadrature wherever the closed form is ill-conditioned.
        # This is rare: on the published grids only a handful of cells qualify,
        # and those which land exactly on a special locus are handled exactly.
        retry = ~np.isfinite(mass) | self._ill_conditioned(x_lower, x_upper, u_lower, u_upper)
        if not np.any(retry):
            return mass
        # Work on flattened copies so that scalar (zero-dimensional) inputs are
        # handled the same way as arrays, then restore the broadcast shape.
        shape = np.shape(mass)
        mass = np.atleast_1d(np.array(mass, dtype=float)).ravel().copy()
        for flat, (xl, xu, ul, uu) in enumerate(
            zip(
                np.atleast_1d(x_lower).ravel(),
                np.atleast_1d(x_upper).ravel(),
                np.atleast_1d(u_lower).ravel(),
                np.atleast_1d(u_upper).ravel(),
                strict=True,
            )
        ):
            if np.atleast_1d(retry).ravel()[flat]:
                mass[flat] = self._cell_mass_quadrature(float(xl), float(xu), float(ul), float(uu))
        return mass.reshape(shape)

    def _ill_conditioned(
        self,
        x_lower: NDArray[np.float64],
        x_upper: NDArray[np.float64],
        u_lower: NDArray[np.float64],
        u_upper: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Whether a cell has a corner too close to a locus the closed form cannot resolve."""
        unsafe = np.zeros(x_lower.shape, dtype=bool)
        for x in (x_lower, x_upper):
            # Small but non-zero radius: the two arctan terms individually
            # diverge as their argument approaches the branch point at +-i, and
            # only their difference is finite.
            unsafe |= (x > 0.0) & (x < _AXIS_TOLERANCE)
            unsafe |= (np.abs(x - 1.0) > 0.0) & (np.abs(x - 1.0) < _SCALE_TOLERANCE)
            for u in (u_lower, u_upper):
                # Unlike the axis and the scale radius, the scale sphere has no
                # exact branch to fall back on, so corners landing exactly on it
                # are flagged too rather than excluded. They are not reliably
                # caught by the non-finite retry either: hypot(x, u) can round to
                # exactly one while x^2 + u^2 - 1, formed differently, comes out
                # a tiny non-zero, giving a finite mass wrong by a factor of a
                # hundred.
                unsafe |= np.abs(np.hypot(x, u) - 1.0) < _SPHERE_TOLERANCE
        return unsafe

    def _cell_mass_quadrature(self, x_lower: float, x_upper: float, u_lower: float, u_upper: float) -> float:
        """Cell mass by exact integration over radius and quadrature over height."""

        def integrand(u: float) -> float:
            return float(
                self._enclosed_annulus(np.hypot(x_upper, u)) - self._enclosed_annulus(np.hypot(x_lower, u))
            )

        # A profile diverging on the axis leaves an integrable logarithmic
        # singularity at u = 0 when the cell reaches the axis; tell the
        # integrator where it is rather than let it hunt.
        points = [0.0] if x_lower == 0.0 and u_lower < 0.0 < u_upper else None
        value, _ = quad(integrand, u_lower, u_upper, points=points, limit=200, epsabs=1.0e-13, epsrel=1.0e-13)
        return 2.0 * np.pi * self.scale_radial**3 * value

    @property
    def optical_depth_radius(self) -> float:
        return self.scale_radial

    @property
    def optical_depth_integral(self) -> float:
        return self._COLUMN_AT_SCALE_RADIUS * self.scale_radial

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scale_radial={self.scale_radial!r})"


class HernquistSpheroid(Spheroid):
    r"""A spherical :math:`\hat\rho = [x(1+x)^3]^{-1}` profile (Hernquist 1990).

    The column along a ray at the scale radius is :math:`4/15` of the scale
    radius, so the normalization for an optical depth :math:`\tau` is
    :math:`\rho_0 = 15\tau / (4 \kappa r_\mathrm{s})`.

    The total mass is :math:`2\pi \rho_0 r_\mathrm{s}^3`.
    """

    _COLUMN_AT_SCALE_RADIUS = 4.0 / 15.0

    def _shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (x * (1.0 + x) ** 3)

    def _enclosed_annulus(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return -0.5 / (1.0 + x) ** 2

    def _antiderivative(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        offset = np.sqrt(np.asarray(x * x - 1.0, dtype=complex))
        result = np.zeros(np.broadcast(x, u).shape, dtype=complex)

        on_axis = x == 0.0
        at_scale = (x == 1.0) & (u != 0.0)
        at_scale_midplane = (x == 1.0) & (u == 0.0)
        generic = (x != 0.0) & (x != 1.0)

        xg, ug, og = x[generic], u[generic], offset[generic]
        radius = np.sqrt(xg * xg + ug * ug)
        result[generic] = 0.5 * (
            ug * (radius - 1.0) / (xg * xg - 1.0) / (xg * xg + ug * ug - 1.0)
            + xg * xg * (np.arctan(ug / radius / og) - np.arctan(ug / og)) / og**3
        )

        # On the axis the general expression reduces to -u / [2(|u| + 1)].
        # The original evaluated an algebraically equivalent form which is 0/0
        # at |u| = 1, and patched that with the constant -1/4 for both signs of
        # u. The antiderivative is odd in u, so +1/4 is correct at u = -1, and
        # the sign was wrong there. Written this way there is no singularity to
        # patch, and the oddness is manifest.
        ua = u[on_axis]
        result[on_axis] = -ua / (2.0 * (np.abs(ua) + 1.0))

        us = u[at_scale]
        root = np.sqrt(1.0 + us * us)
        result[at_scale] = (-2.0 - 4.0 * us**2 - 2.0 * us**4 + 2.0 * root + 3.0 * us**2 * root) / (
            6.0 * us**3 * root
        )
        result[at_scale_midplane] = 0.0

        return np.real(result)


class JaffeSpheroid(Spheroid):
    r"""A spherical :math:`\hat\rho = [x^2(1+x)^2]^{-1}` profile (Jaffe 1983).

    The column along a ray at the scale radius is :math:`\pi - 8/3` of the scale
    radius, so the normalization for an optical depth :math:`\tau` is
    :math:`\rho_0 = \tau / [(\pi - 8/3) \kappa r_\mathrm{s}]`.

    The total mass is :math:`4\pi \rho_0 r_\mathrm{s}^3`.
    """

    _COLUMN_AT_SCALE_RADIUS = np.pi - 8.0 / 3.0

    def _shape(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        return 1.0 / (x**2 * (1.0 + x) ** 2)

    def _enclosed_annulus(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        with np.errstate(divide="ignore"):
            return np.log(x / (1.0 + x)) + 1.0 / (1.0 + x)

    def _antiderivative(self, x: NDArray[np.float64], u: NDArray[np.float64]) -> NDArray[np.float64]:
        offset = np.sqrt(np.asarray(x * x - 1.0, dtype=complex))
        result = np.zeros(np.broadcast(x, u).shape, dtype=complex)

        on_axis_midplane = (x == 0.0) & (u == 0.0)
        on_axis = (x == 0.0) & (u != 0.0)
        at_scale = (x == 1.0) & (u != 0.0)
        at_scale_midplane = (x == 1.0) & (u == 0.0)
        generic = (x != 0.0) & (x != 1.0)

        xg, ug, og = x[generic], u[generic], offset[generic]
        radius = np.sqrt(xg * xg + ug * ug)
        result[generic] = xg * (
            np.arctan(ug / xg) + xg * (-np.arctan(ug / og) + np.arctan(ug / og / radius)) / og
        ) + 0.5 * ug * (np.log(xg * xg + ug * ug) - 2.0 * np.log(1.0 + radius))

        # On the axis the general expression reduces to u ln[|u| / (1 + |u|)],
        # which is regular everywhere except u = 0. The original special-cased
        # |u| = 1 to the constant -ln 2 for both signs of u, but the
        # antiderivative is odd in u, so +ln 2 is correct at u = -1. The special
        # case is not needed at all.
        ua = u[on_axis]
        result[on_axis] = ua * np.log(np.abs(ua) / (1.0 + np.abs(ua)))
        result[on_axis_midplane] = 0.0

        us = u[at_scale]
        root = np.sqrt(1.0 + us * us)
        result[at_scale] = (
            np.arctan(us)
            + 0.5 * us * np.log(1.0 + us * us)
            - (-1.0 + root + us * us * np.log(1.0 + root)) / us
        )
        result[at_scale_midplane] = 0.0

        return np.real(result)
