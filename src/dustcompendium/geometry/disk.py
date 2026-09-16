r"""Disk profiles: exponential in radius, with a choice of vertical structure.

The disk is separable,

.. math::

    \hat\rho(R, z) = \mathrm{e}^{-R/R_\mathrm{d}} \, f(z),

so its cell masses factorize into a radial and a vertical integral. The
vertical structure :math:`f`, with :math:`f(0) = 1`, is supplied by a
:class:`VerticalStructure`; adding a new one means adding a class with a shape
and its antiderivative, and registering it.
"""

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .base import Profile

__all__ = [
    "ExponentialDisk",
    "ExponentialVertical",
    "SechSquaredVertical",
    "VerticalStructure",
]


class VerticalStructure(ABC):
    """The vertical part of a separable disk profile."""

    def __init__(self, scale_height: float) -> None:
        if scale_height <= 0.0:
            raise ValueError(f"scale height must be positive, got {scale_height}")
        self.scale_height = float(scale_height)

    @abstractmethod
    def shape(self, height: ArrayLike) -> NDArray[np.float64]:
        """The vertical shape :math:`f(z)`, normalized so that :math:`f(0) = 1`."""

    @abstractmethod
    def column(self, height: ArrayLike) -> NDArray[np.float64]:
        r"""The antiderivative :math:`F(z) = \int_0^z f(\zeta)\,\mathrm{d}\zeta`.

        Defined with :math:`F(0) = 0` and carrying the sign of :math:`z`, so
        that :math:`\int_{z_0}^{z_1} f = F(z_1) - F(z_0)` holds for any
        interval, including one straddling the midplane. Folding about
        :math:`z = 0` instead -- taking absolute values of the limits -- gives
        zero for a cell symmetric about the midplane, which is where most of a
        disk's mass lies.
        """

    @property
    def total_column(self) -> float:
        r""":math:`\int_{-\infty}^{\infty} f(z)\,\mathrm{d}z`."""
        return 2.0 * self._half_column

    @property
    @abstractmethod
    def _half_column(self) -> float:
        r""":math:`\int_0^{\infty} f(z)\,\mathrm{d}z`."""

    def __repr__(self) -> str:
        return f"{type(self).__name__}(scale_height={self.scale_height!r})"


class ExponentialVertical(VerticalStructure):
    r"""Exponential vertical structure, :math:`f(z) = \mathrm{e}^{-|z|/h_z}`."""

    def shape(self, height: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(height, dtype=float)
        return np.exp(-np.abs(z) / self.scale_height)

    def column(self, height: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(height, dtype=float)
        return np.sign(z) * self.scale_height * (1.0 - np.exp(-np.abs(z) / self.scale_height))

    @property
    def _half_column(self) -> float:
        return self.scale_height


class SechSquaredVertical(VerticalStructure):
    r"""Isothermal sheet vertical structure, :math:`f(z) = \mathrm{sech}^2(z/h_z)`.

    Evaluated through :func:`numpy.cosh`, which overflows to infinity for large
    arguments and so returns zero density rather than raising.
    """

    def shape(self, height: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(height, dtype=float)
        with np.errstate(over="ignore"):
            return np.where(
                np.abs(z) / self.scale_height < 350.0,
                1.0 / np.cosh(np.clip(z / self.scale_height, -350.0, 350.0)) ** 2,
                0.0,
            )

    def column(self, height: ArrayLike) -> NDArray[np.float64]:
        z = np.asarray(height, dtype=float)
        return self.scale_height * np.tanh(z / self.scale_height)

    @property
    def _half_column(self) -> float:
        return self.scale_height


class ExponentialDisk(Profile):
    r"""A disk, exponential in radius with a given vertical structure.

    .. math::

        \hat\rho(R, z) = \mathrm{e}^{-R/R_\mathrm{d}} \, f(z).

    The optical depth is defined through the centre, viewed face-on. Both
    vertical structures give a central column of :math:`2 h_z`, so a disk's
    normalization depends on its vertical scale height but not on which of the
    two structures it uses -- a coincidence of these two profiles, not a general
    rule, and the reason :attr:`optical_depth_integral` asks the vertical
    structure rather than assuming it.

    Parameters
    ----------
    scale_radial
        The radial scale length :math:`R_\mathrm{d}`, positive.
    vertical
        The vertical structure, which carries its own scale height.
    """

    def __init__(self, scale_radial: float, vertical: VerticalStructure) -> None:
        if scale_radial <= 0.0:
            raise ValueError(f"radial scale length must be positive, got {scale_radial}")
        self.scale_radial = float(scale_radial)
        self.vertical = vertical

    def density(self, radius: ArrayLike, height: ArrayLike) -> NDArray[np.float64]:
        r = np.asarray(radius, dtype=float)
        return np.exp(-r / self.scale_radial) * self.vertical.shape(height)

    def _radial_column(self, radius: ArrayLike) -> NDArray[np.float64]:
        r"""The antiderivative :math:`\int_0^R \mathrm{e}^{-R'/R_\mathrm{d}} R'\,\mathrm{d}R'`."""
        r = np.asarray(radius, dtype=float)
        scale = self.scale_radial
        return scale * (scale - (r + scale) * np.exp(-r / scale))

    def cell_mass(
        self,
        radius_lower: ArrayLike,
        radius_upper: ArrayLike,
        height_lower: ArrayLike,
        height_upper: ArrayLike,
    ) -> NDArray[np.float64]:
        radial = self._radial_column(radius_upper) - self._radial_column(radius_lower)
        vertical = self.vertical.column(height_upper) - self.vertical.column(height_lower)
        return 2.0 * np.pi * radial * vertical

    @property
    def extent_radial(self) -> float:
        return self.scale_radial

    @property
    def extent_vertical(self) -> float:
        return self.vertical.scale_height

    @property
    def optical_depth_radius(self) -> float:
        return 0.0

    @property
    def optical_depth_integral(self) -> float:
        return self.vertical.total_column

    def __repr__(self) -> str:
        return f"ExponentialDisk(scale_radial={self.scale_radial!r}, vertical={self.vertical!r})"
