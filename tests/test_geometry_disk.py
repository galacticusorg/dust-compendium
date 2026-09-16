"""Tests for the disk profiles."""

import numpy as np
import pytest

from dustcompendium.geometry import (
    ExponentialDisk,
    ExponentialVertical,
    SechSquaredVertical,
)

from .reference import separable_cell_mass

SCALE_RADIAL = 1.3
SCALE_HEIGHT = 0.137

VERTICALS = [
    pytest.param(ExponentialVertical(SCALE_HEIGHT), id="exponential"),
    pytest.param(SechSquaredVertical(SCALE_HEIGHT), id="sechSquared"),
]

# Cells chosen to exercise the midplane, the axis, and the far tail. The
# straddling and symmetric cases are the ones the original implementation got
# wrong; see test_symmetric_cell_is_not_empty.
CELLS = [
    pytest.param(0.3, 0.7, 0.04, 0.09, id="generic"),
    pytest.param(0.0, 0.5, 0.04, 0.09, id="touches-axis"),
    pytest.param(0.3, 0.7, -0.04, 0.09, id="straddles-midplane"),
    pytest.param(0.3, 0.7, -0.05, 0.05, id="symmetric-about-midplane"),
    pytest.param(0.3, 0.7, -0.09, -0.04, id="below-midplane"),
    pytest.param(0.0, 0.05, 0.0, 0.005, id="innermost"),
    pytest.param(8.0, 13.0, 1.0, 2.0, id="far-tail"),
]


@pytest.mark.parametrize("vertical", VERTICALS)
class TestVerticalStructure:
    def test_shape_is_unity_at_the_midplane(self, vertical):
        assert vertical.shape(0.0) == pytest.approx(1.0)

    def test_column_is_the_antiderivative_of_the_shape(self, vertical):
        """Differentiate the column numerically and recover the shape.

        The midplane is excluded: the exponential structure has a corner there,
        so a central difference straddling it is not a valid derivative
        estimate. :meth:`test_shape_is_unity_at_the_midplane` covers that point.
        """
        height = np.array([-0.4, -0.05, 0.05, 0.4])
        step = 1.0e-6
        derivative = (vertical.column(height + step) - vertical.column(height - step)) / (2.0 * step)
        np.testing.assert_allclose(derivative, vertical.shape(height), rtol=1.0e-8)

    def test_column_vanishes_at_the_midplane(self, vertical):
        assert vertical.column(0.0) == pytest.approx(0.0, abs=1.0e-15)

    def test_column_is_odd_in_height(self, vertical):
        height = np.array([0.01, 0.13, 1.7])
        np.testing.assert_allclose(vertical.column(-height), -vertical.column(height), rtol=1.0e-13)

    def test_total_column_matches_quadrature(self, vertical):
        from scipy.integrate import quad

        expected, _ = quad(vertical.shape, -np.inf, np.inf, epsabs=1.0e-13)
        assert vertical.total_column == pytest.approx(expected, rel=1.0e-10)

    def test_negative_scale_height_is_rejected(self, vertical):
        with pytest.raises(ValueError, match="scale height must be positive"):
            type(vertical)(-1.0)


@pytest.mark.parametrize("vertical", VERTICALS)
class TestExponentialDisk:
    @pytest.mark.parametrize(("r_lower", "r_upper", "z_lower", "z_upper"), CELLS)
    def test_cell_mass_matches_quadrature(self, vertical, r_lower, r_upper, z_lower, z_upper):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        expected = separable_cell_mass(
            lambda radius: np.exp(-radius / SCALE_RADIAL),
            vertical.shape,
            r_lower,
            r_upper,
            z_lower,
            z_upper,
        )
        assert float(disk.cell_mass(r_lower, r_upper, z_lower, z_upper)) == pytest.approx(
            expected, rel=1.0e-10
        )

    def test_symmetric_cell_is_not_empty(self, vertical):
        """Regression: the original vertical factor folded about the midplane.

        ``hyperionBuildModel.py`` took absolute values of both limits before
        differencing, so a cell symmetric about ``z = 0`` returned exactly zero.
        The published grids put ``z = 0`` on a cell wall and so never hit it, but
        an even number of vertical walls would have, and silently: the midplane
        cells hold most of a disk's stellar mass.
        """
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        mass = float(disk.cell_mass(0.3, 0.7, -0.05, 0.05))
        assert mass > 0.0
        legacy = abs(abs(vertical.column(0.05)) - abs(vertical.column(-0.05)))
        assert legacy == pytest.approx(0.0, abs=1.0e-15)

    def test_cell_mass_is_additive_in_height(self, vertical):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        whole = disk.cell_mass(0.3, 0.7, -0.08, 0.11)
        parts = sum(
            disk.cell_mass(0.3, 0.7, lower, upper)
            for lower, upper in [(-0.08, -0.02), (-0.02, 0.0), (0.0, 0.04), (0.04, 0.11)]
        )
        assert float(parts) == pytest.approx(float(whole), rel=1.0e-12)

    def test_cell_mass_is_additive_in_radius(self, vertical):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        whole = disk.cell_mass(0.0, 2.0, 0.01, 0.2)
        parts = sum(
            disk.cell_mass(lower, upper, 0.01, 0.2) for lower, upper in [(0.0, 0.3), (0.3, 0.9), (0.9, 2.0)]
        )
        assert float(parts) == pytest.approx(float(whole), rel=1.0e-12)

    def test_total_mass_matches_the_analytic_value(self, vertical):
        r"""Total mass is :math:`2\pi R_\mathrm{d}^2 \int f(z)\,\mathrm{d}z`."""
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        total = float(disk.cell_mass(0.0, 400.0 * SCALE_RADIAL, -400.0, 400.0))
        expected = 2.0 * np.pi * SCALE_RADIAL**2 * vertical.total_column
        assert total == pytest.approx(expected, rel=1.0e-10)

    def test_optical_depth_is_defined_through_the_centre(self, vertical):
        assert ExponentialDisk(SCALE_RADIAL, vertical).optical_depth_radius == 0.0

    def test_normalization_reproduces_the_requested_optical_depth(self, vertical):
        """Integrate the normalized density along the ray and recover tau."""
        from scipy.integrate import quad

        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        optical_depth, opacity = 3.7, 1.9
        normalization = disk.density_normalization(optical_depth, opacity)
        column, _ = quad(
            lambda z: normalization * float(disk.density(disk.optical_depth_radius, z)),
            -np.inf,
            np.inf,
            epsabs=1.0e-13,
        )
        assert opacity * column == pytest.approx(optical_depth, rel=1.0e-9)

    def test_both_vertical_structures_share_a_normalization(self, vertical):
        """Both have a central column of 2 h_z, as the original code assumed."""
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        assert disk.optical_depth_integral == pytest.approx(2.0 * SCALE_HEIGHT, rel=1.0e-13)

    def test_density_is_separable(self, vertical):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        radius, height = 0.7, 0.05
        assert float(disk.density(radius, height)) == pytest.approx(
            float(disk.density(radius, 0.0)) * float(disk.density(0.0, height))
        )

    def test_density_broadcasts(self, vertical):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        radius = np.linspace(0.0, 2.0, 5).reshape(-1, 1)
        height = np.linspace(-0.2, 0.2, 3).reshape(1, -1)
        assert disk.density(radius, height).shape == (5, 3)

    def test_sech_squared_does_not_overflow_far_from_the_midplane(self, vertical):
        disk = ExponentialDisk(SCALE_RADIAL, vertical)
        value = disk.density(0.0, 1.0e6)
        assert np.isfinite(value) and value == pytest.approx(0.0)


def test_negative_radial_scale_is_rejected():
    with pytest.raises(ValueError, match="radial scale length must be positive"):
        ExponentialDisk(-1.0, SechSquaredVertical(SCALE_HEIGHT))


@pytest.mark.parametrize(("optical_depth", "opacity"), [(-1.0, 1.0), (1.0, 0.0), (1.0, -1.0)])
def test_invalid_normalization_arguments_are_rejected(optical_depth, opacity):
    disk = ExponentialDisk(SCALE_RADIAL, SechSquaredVertical(SCALE_HEIGHT))
    with pytest.raises(ValueError):
        disk.density_normalization(optical_depth, opacity)
