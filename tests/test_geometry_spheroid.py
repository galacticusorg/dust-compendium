"""Tests for the spheroid profiles.

Several of these are regressions against defects in the original
``hyperionBuildModel.py`` implementation. All four were latent -- the published
tabulations are unaffected, because their grids happen to avoid every one -- but
each would have produced silently wrong masses on a slightly different grid.
"""

from itertools import pairwise

import numpy as np
import pytest

from dustcompendium.geometry import HernquistSpheroid, JaffeSpheroid

from .reference import HERNQUIST, JAFFE, column_along_ray, spherical_cell_mass

SCALE = 1.7

PROFILES = [
    pytest.param(HernquistSpheroid, HERNQUIST, id="hernquist"),
    pytest.param(JaffeSpheroid, JAFFE, id="jaffe"),
]

# In units of the scale radius. The last four sit on, or just off, the loci
# where the closed form is singular or ill-conditioned.
CELLS = [
    pytest.param(0.3, 0.7, 0.4, 0.9, id="generic"),
    pytest.param(0.0, 0.5, 0.4, 0.9, id="touches-axis"),
    pytest.param(0.8, 1.3, 0.4, 0.9, id="spans-scale-radius"),
    pytest.param(0.3, 0.7, -0.4, 0.9, id="straddles-midplane"),
    pytest.param(0.3, 0.7, -0.5, 0.5, id="symmetric-about-midplane"),
    pytest.param(3.0, 5.0, 2.0, 4.0, id="far-out"),
    pytest.param(0.6, 1.4, 0.8, 1.1, id="corner-on-sphere"),
    pytest.param(0.6, 1.4, 0.8 + 1.0e-12, 1.1, id="corner-just-off-sphere"),
    pytest.param(1.0e-13, 0.4, 0.3, 0.8, id="wall-just-off-axis"),
    pytest.param(1.0 + 1.0e-13, 1.4, 0.3, 0.8, id="wall-just-off-scale-radius"),
    pytest.param(0.0, 0.4, -1.0, -0.6, id="on-axis-below-scale-radius"),
    pytest.param(0.0, 0.4, -1.0, 1.0, id="on-axis-spanning-scale-radius"),
    pytest.param(0.0, 0.5, -2.0, -1.0, id="on-axis-beyond-scale-radius"),
    pytest.param(0.6, 1.4, -0.8, -0.5, id="corner-on-sphere-below-midplane"),
]


@pytest.mark.parametrize(("profile_class", "shape"), PROFILES)
class TestSpheroid:
    @pytest.mark.parametrize(("x_lower", "x_upper", "u_lower", "u_upper"), CELLS)
    def test_cell_mass_matches_quadrature(self, profile_class, shape, x_lower, x_upper, u_lower, u_upper):
        profile = profile_class(SCALE)
        got = float(profile.cell_mass(x_lower * SCALE, x_upper * SCALE, u_lower * SCALE, u_upper * SCALE))
        expected = spherical_cell_mass(
            shape, SCALE, x_lower * SCALE, x_upper * SCALE, u_lower * SCALE, u_upper * SCALE
        )
        assert got == pytest.approx(expected, rel=1.0e-9)

    @pytest.mark.parametrize("x", [0.0, 0.3, 1.0, 2.0])
    def test_antiderivative_is_odd_in_height(self, profile_class, shape, x):
        """The profiles are symmetric about the midplane, so the antiderivative is odd.

        Regression: the original assigned a constant to the ``R = 0``,
        ``|z| = r_s`` corner without regard to the sign of ``z``, giving the
        wrong sign below the midplane. Unlike the other defects this one is not
        latent -- ``z = -r_s`` falls exactly on a vertical wall whenever the
        spheroid scale radius sets the vertical extent of the grid, which it
        does for every published spheroid larger than the disk scale height.
        """
        profile = profile_class(1.0)
        heights = np.array([0.1, 0.5, 1.0, 2.5])
        radii = np.full_like(heights, x)
        above = profile._antiderivative(radii, heights)
        below = profile._antiderivative(radii, -heights)
        np.testing.assert_allclose(above, -below, atol=1.0e-14)

    def test_cell_mass_below_the_midplane_at_the_scale_radius(self, profile_class, shape):
        """Regression for the sign defect, at the corner the published grids hit."""
        profile = profile_class(1.0)
        cell = (0.0, 0.4, -1.0, -0.6)
        got = float(profile.cell_mass(*cell))
        assert got > 0.0
        assert got == pytest.approx(spherical_cell_mass(shape, 1.0, *cell), rel=1.0e-9)
        mirrored = float(profile.cell_mass(0.0, 0.4, 0.6, 1.0))
        assert got == pytest.approx(mirrored, rel=1.0e-12)

    def test_cell_mass_is_finite_on_the_scale_sphere(self, profile_class, shape):
        """Regression: the generic closed form evaluates to 0/0 on ``r = r_s``.

        The original implementation special-cased ``R = 0`` and ``R = r_s`` but
        not the rest of the sphere, where both expressions divide by
        ``x^2 + u^2 - 1``. A corner landing on it, such as ``(0.6, 0.8) r_s``,
        returned NaN, and one landing within about ``1e-9`` of it lost most of
        its precision.
        """
        profile = profile_class(1.0)
        for x in (0.0, 0.28, 0.6, 0.8, 1.0):
            for sign in (1.0, -1.0):
                u = sign * np.sqrt(max(1.0 - x * x, 0.0))
                cell = (x, x + 0.4, u, u + 0.3)
                mass = float(profile.cell_mass(*cell))
                assert np.isfinite(mass), f"not finite with a corner at ({x}, {u})"
                assert mass > 0.0
                assert mass == pytest.approx(spherical_cell_mass(shape, 1.0, *cell), rel=1.0e-9), (
                    f"corner at ({x}, {u})"
                )

    def test_cell_mass_is_accurate_at_every_corner_on_the_scale_sphere(self, profile_class, shape):
        """Sweep the sphere, rather than sample it at a few convenient points.

        A corner can land on the sphere in two inequivalent ways: with
        ``x^2 + u^2 - 1`` exactly zero, which makes the closed form NaN and
        takes the non-finite retry, or with it merely tiny, which returns a
        finite mass wrong by up to a factor of a hundred. Only the first is hit
        by round numbers, so the sweep is what finds the second.
        """
        profile = profile_class(1.0)
        for x in np.linspace(0.005, 0.995, 60):
            u = np.sqrt(1.0 - x * x)
            cell = (float(x), float(x) + 0.4, float(u), float(u) + 0.3)
            assert float(profile.cell_mass(*cell)) == pytest.approx(
                spherical_cell_mass(shape, 1.0, *cell), rel=1.0e-9
            ), f"corner at ({x}, {u}), denominator {x * x + u * u - 1.0!r}"

    def test_cell_mass_is_accurate_approaching_the_scale_sphere(self, profile_class, shape):
        """Sweep a corner towards the sphere and stay accurate throughout."""
        profile = profile_class(1.0)
        base = np.sqrt(1.0 - 0.36)
        for offset in [0.0, 1.0e-14, 1.0e-12, 1.0e-9, 1.0e-6, 1.0e-3]:
            cell = (0.6, 1.4, base + offset, base + offset + 0.3)
            got = float(profile.cell_mass(*cell))
            expected = spherical_cell_mass(shape, 1.0, *cell)
            assert got == pytest.approx(expected, rel=1.0e-9), f"offset {offset}"

    def test_cell_mass_is_accurate_approaching_the_axis(self, profile_class, shape):
        """Regression: a small but non-zero inner radius returned NaN."""
        profile = profile_class(1.0)
        for radius in [0.0, 1.0e-14, 1.0e-11, 1.0e-8, 1.0e-5, 1.0e-2]:
            cell = (radius, 0.4, 0.3, 0.8)
            got = float(profile.cell_mass(*cell))
            expected = spherical_cell_mass(shape, 1.0, *cell)
            assert got == pytest.approx(expected, rel=1.0e-9), f"radius {radius}"

    def test_cell_mass_is_accurate_approaching_the_scale_radius(self, profile_class, shape):
        """Regression: Hernquist lost all precision within ~1e-5 of ``R = r_s``."""
        profile = profile_class(1.0)
        for offset in [0.0, 1.0e-13, 1.0e-11, 1.0e-9, 1.0e-7, 1.0e-5, 1.0e-3]:
            for cell in [(1.0 + offset, 1.4, 0.3, 0.8), (0.6, 1.0 - offset, 0.3, 0.8)]:
                got = float(profile.cell_mass(*cell))
                expected = spherical_cell_mass(shape, 1.0, *cell)
                assert got == pytest.approx(expected, rel=1.0e-9), f"{cell}"

    def test_symmetric_cell_is_not_empty(self, profile_class, shape):
        """Regression: the original combined corners through nested absolute values.

        ``abs(abs(F11 - F01) - abs(F10 - F00))`` returns zero for a cell
        symmetric about the midplane, instead of the plain two-dimensional
        difference. The published grids put ``z = 0`` on a wall, so no cell
        straddled it.
        """
        profile = profile_class(1.0)
        antiderivative = profile._antiderivative
        corners = {
            (x, u): float(antiderivative(np.array([x]), np.array([u]))[0])
            for x in (0.3, 0.7)
            for u in (-0.5, 0.5)
        }
        legacy = abs(
            abs(corners[(0.7, 0.5)] - corners[(0.3, 0.5)]) - abs(corners[(0.7, -0.5)] - corners[(0.3, -0.5)])
        )
        assert legacy == pytest.approx(0.0, abs=1.0e-15)
        assert float(profile.cell_mass(0.3, 0.7, -0.5, 0.5)) > 0.0

    def test_cell_mass_is_additive(self, profile_class, shape):
        profile = profile_class(SCALE)
        whole = float(profile.cell_mass(0.4, 3.0, -1.0, 2.0))
        parts = 0.0
        radii = [0.4, 0.9, 1.7, 3.0]
        heights = [-1.0, -0.3, 0.0, 0.8, 2.0]
        for lower, upper in pairwise(radii):
            for low, high in pairwise(heights):
                parts += float(profile.cell_mass(lower, upper, low, high))
        assert parts == pytest.approx(whole, rel=1.0e-9)

    def test_cell_mass_is_symmetric_about_the_midplane(self, profile_class, shape):
        profile = profile_class(SCALE)
        above = float(profile.cell_mass(0.4, 1.1, 0.3, 0.9))
        below = float(profile.cell_mass(0.4, 1.1, -0.9, -0.3))
        assert above == pytest.approx(below, rel=1.0e-12)

    def test_normalization_reproduces_the_requested_optical_depth(self, profile_class, shape):
        profile = profile_class(SCALE)
        optical_depth, opacity = 2.3, 0.7
        normalization = profile.density_normalization(optical_depth, opacity)
        column = column_along_ray(
            lambda radius, height: normalization * float(profile.density(radius, height)),
            profile.optical_depth_radius,
        )
        assert opacity * column == pytest.approx(optical_depth, rel=1.0e-9)

    def test_optical_depth_is_defined_at_the_scale_radius(self, profile_class, shape):
        """The central column diverges, so the ray is offset to the scale radius."""
        profile = profile_class(SCALE)
        assert profile.optical_depth_radius == pytest.approx(SCALE)

    def test_density_diverges_at_the_centre(self, profile_class, shape):
        profile = profile_class(SCALE)
        assert np.isinf(float(profile.density(0.0, 0.0)))

    def test_cell_mass_scales_with_the_cube_of_the_scale_radius(self, profile_class, shape):
        """Masses at fixed shape depend on the scale radius only through its cube."""
        cell = (0.3, 0.9, 0.2, 0.7)
        unit = float(profile_class(1.0).cell_mass(*cell))
        scaled = float(profile_class(SCALE).cell_mass(*(value * SCALE for value in cell)))
        assert scaled == pytest.approx(unit * SCALE**3, rel=1.0e-10)

    def test_negative_scale_radius_is_rejected(self, profile_class, shape):
        with pytest.raises(ValueError, match="scale radius must be positive"):
            profile_class(-1.0)


def test_hernquist_column_at_the_scale_radius():
    """The normalization the original code hard-coded as ``15/4``."""
    profile = HernquistSpheroid(1.0)
    expected = column_along_ray(lambda radius, height: float(profile.density(radius, height)), 1.0)
    assert profile.optical_depth_integral == pytest.approx(expected, rel=1.0e-10)
    assert profile.optical_depth_integral == pytest.approx(4.0 / 15.0, rel=1.0e-13)
    assert profile.density_normalization(1.0, 1.0) == pytest.approx(15.0 / 4.0, rel=1.0e-13)


def test_jaffe_column_at_the_scale_radius():
    """The normalization the original code hard-coded as ``1/(pi - 8/3)``."""
    profile = JaffeSpheroid(1.0)
    expected = column_along_ray(lambda radius, height: float(profile.density(radius, height)), 1.0)
    assert profile.optical_depth_integral == pytest.approx(expected, rel=1.0e-10)
    assert profile.optical_depth_integral == pytest.approx(np.pi - 8.0 / 3.0, rel=1.0e-13)
    assert profile.density_normalization(1.0, 1.0) == pytest.approx(1.0 / (np.pi - 8.0 / 3.0), rel=1.0e-13)


@pytest.mark.parametrize(
    ("profile_class", "total"),
    [(HernquistSpheroid, 2.0 * np.pi), (JaffeSpheroid, 4.0 * np.pi)],
)
def test_total_mass_matches_the_analytic_value(profile_class, total):
    r"""Hernquist integrates to :math:`2\pi\rho_0 r_\mathrm{s}^3`, Jaffe to :math:`4\pi`."""
    profile = profile_class(SCALE)
    far = 1.0e7 * SCALE
    got = float(profile.cell_mass(0.0, far, -far, far))
    assert got == pytest.approx(total * SCALE**3, rel=1.0e-5)
