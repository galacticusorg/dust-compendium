r"""The density profile interface.

A :class:`Profile` describes the *shape* of a density distribution, separately
from its normalization. Every profile is written as

.. math::

    \rho(R, z) = \rho_0 \, \hat\rho(R, z),

where :math:`\hat\rho` is the dimensionless shape returned by
:meth:`Profile.density` and :math:`\rho_0` is a normalization with units of
density. The two are kept apart because the models are normalized by optical
depth rather than by mass: :math:`\rho_0` is whatever reproduces a requested
optical depth for a given opacity, which is what
:meth:`Profile.density_normalization` computes.

What :math:`\rho_0` *means* differs between profiles, and the difference
matters. For a disk it is the density at the centre. For the Hernquist and
Jaffe spheroids the density diverges at the centre, so :math:`\rho_0` is only a
scale density and the optical depth through the centre is infinite -- which is
why those profiles define their optical depth along a ray offset from the axis.
See :attr:`Profile.optical_depth_radius`.

Lengths are in whatever unit the scale lengths were given in; every method is
consistent in that unit, and nothing here assumes a particular one.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

__all__ = ["Profile"]


class Profile(ABC):
    """A density profile with cylindrical symmetry and reflection symmetry about the midplane."""

    @abstractmethod
    def density(self, radius: ArrayLike, height: ArrayLike) -> NDArray[np.float64]:
        r"""The dimensionless density shape :math:`\hat\rho` at cylindrical coordinates.

        Parameters
        ----------
        radius
            Cylindrical radius :math:`R`, non-negative.
        height
            Height above the midplane :math:`z`, of either sign.

        Returns
        -------
        The shape function, broadcast over the inputs. Multiply by a
        normalization from :meth:`density_normalization` to obtain a density.
        Profiles which diverge at the origin return ``inf`` there rather than
        raising.
        """

    @abstractmethod
    def cell_mass(
        self,
        radius_lower: ArrayLike,
        radius_upper: ArrayLike,
        height_lower: ArrayLike,
        height_upper: ArrayLike,
    ) -> NDArray[np.float64]:
        r"""The mass of an annular cell, per unit normalization.

        Integrates :math:`\hat\rho` over the full azimuth and over
        :math:`R \in [R_\mathrm{lower}, R_\mathrm{upper}]`,
        :math:`z \in [z_\mathrm{lower}, z_\mathrm{upper}]`:

        .. math::

            \frac{M}{\rho_0}
            = 2\pi \int_{z_\mathrm{lower}}^{z_\mathrm{upper}}
                   \int_{R_\mathrm{lower}}^{R_\mathrm{upper}}
                   \hat\rho(R, z) \, R \, \mathrm{d}R \, \mathrm{d}z.

        Cells straddling the midplane are handled correctly: the integral is
        signed in :math:`z`, not folded about :math:`z = 0`.

        Returns
        -------
        Mass per unit normalization, with units of length cubed, broadcast over
        the inputs.
        """

    @property
    @abstractmethod
    def optical_depth_radius(self) -> float:
        r"""Cylindrical radius of the ray along which optical depth is defined.

        Optical depth is always measured along a ray parallel to the symmetry
        axis, running from :math:`z = -\infty` to :math:`z = +\infty` at this
        radius. It is zero -- a ray through the centre -- for profiles with a
        finite central column, and the scale radius for profiles whose central
        column diverges.
        """

    @property
    @abstractmethod
    def optical_depth_integral(self) -> float:
        r"""The column of :math:`\hat\rho` along the ray, with units of length.

        .. math::

            I = \int_{-\infty}^{\infty}
                \hat\rho(R_\tau, z) \, \mathrm{d}z,

        evaluated at :math:`R_\tau =` :attr:`optical_depth_radius`. The optical
        depth of the fully normalized profile is then
        :math:`\tau = \kappa \rho_0 I` for an opacity :math:`\kappa` per unit
        mass.
        """

    def density_normalization(self, optical_depth: float, opacity: float) -> float:
        r"""The normalization :math:`\rho_0` giving a requested optical depth.

        Inverts :math:`\tau = \kappa \rho_0 I`, so that a profile scaled by the
        result has optical depth ``optical_depth`` along
        :attr:`optical_depth_radius`.

        Parameters
        ----------
        optical_depth
            The optical depth to reproduce, non-negative.
        opacity
            Opacity to extinction per unit mass of the attenuating material,
            :math:`\kappa`, in units inverse to those of a column density. Must
            be positive.

        Raises
        ------
        ValueError
            If ``opacity`` is not positive, or ``optical_depth`` is negative.
        """
        if opacity <= 0.0:
            raise ValueError(f"opacity must be positive, got {opacity}")
        if optical_depth < 0.0:
            raise ValueError(f"optical depth must be non-negative, got {optical_depth}")
        return optical_depth / (opacity * self.optical_depth_integral)
