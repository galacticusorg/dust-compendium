"""Whole-grid checks, on the grid the published tabulations were computed on."""

import numpy as np
import pytest

from dustcompendium.geometry import (
    ExponentialDisk,
    ExponentialVertical,
    HernquistSpheroid,
    JaffeSpheroid,
    SechSquaredVertical,
)

CUT_OFF = 10.0
SCALE_RADIAL = 1.0
SCALE_HEIGHT = 0.137


def published_grid():
    """The cell walls from ``hyperionBuildModel.py`` lines 107-111.

    A logarithmic radial grid with a wall added at the axis, and a linear
    vertical grid. The vertical grid has an odd number of walls, which is why
    ``z = 0`` falls on a wall and no cell straddles the midplane -- the reason
    the original folded vertical integral never showed itself.
    """
    radial = np.hstack(
        [0.0, np.logspace(np.log10(1.0e-2 * SCALE_RADIAL), np.log10(CUT_OFF * SCALE_RADIAL), 100)]
    )
    vertical = np.linspace(-CUT_OFF * SCALE_HEIGHT, CUT_OFF * SCALE_HEIGHT, 101)
    return radial, vertical


def cell_walls():
    radial, vertical = published_grid()
    return np.broadcast_arrays(radial[:-1, None], radial[1:, None], vertical[None, :-1], vertical[None, 1:])


PROFILES = [
    pytest.param(HernquistSpheroid(SCALE_RADIAL), id="hernquist"),
    pytest.param(JaffeSpheroid(SCALE_RADIAL), id="jaffe"),
    pytest.param(ExponentialDisk(SCALE_RADIAL, SechSquaredVertical(SCALE_HEIGHT)), id="disk-sechSquared"),
    pytest.param(ExponentialDisk(SCALE_RADIAL, ExponentialVertical(SCALE_HEIGHT)), id="disk-exponential"),
]


def test_the_published_vertical_grid_puts_the_midplane_on_a_wall():
    """Why the folded vertical integral never produced a wrong published table."""
    _, vertical = published_grid()
    assert np.any(vertical == 0.0)
    assert not np.any((vertical[:-1] < 0.0) & (vertical[1:] > 0.0))


@pytest.mark.parametrize("profile", PROFILES)
class TestOnThePublishedGrid:
    def test_every_cell_mass_is_finite_and_non_negative(self, profile):
        mass = profile.cell_mass(*cell_walls())
        assert np.all(np.isfinite(mass))
        assert np.all(mass >= 0.0)

    def test_masses_sum_to_the_mass_of_the_enclosing_cell(self, profile):
        """The grid tiles its own bounding box, so the cells must telescope."""
        radial, vertical = published_grid()
        total = float(profile.cell_mass(radial[0], radial[-1], vertical[0], vertical[-1]))
        assert float(np.sum(profile.cell_mass(*cell_walls()))) == pytest.approx(total, rel=1.0e-9)

    def test_mass_is_symmetric_about_the_midplane(self, profile):
        """Compared against the total, which is what the cell masses are used for.

        A cell mass is a difference of antiderivatives, so the smallest cells --
        those hugging the axis, holding some 1e-5 of the peak cell -- lose
        relative precision to cancellation, by around 1e-8. What matters is the
        emission probability, the mass relative to the total, and that is
        symmetric to better than 1e-11.
        """
        mass = profile.cell_mass(*cell_walls())
        np.testing.assert_allclose(mass, mass[:, ::-1], rtol=1.0e-6, atol=1.0e-11 * mass.sum())


@pytest.mark.parametrize("scale_radius", [0.137, 0.1585, 0.5, 1.0, 10.0])
@pytest.mark.parametrize("profile_class", [HernquistSpheroid, JaffeSpheroid])
def test_emission_is_not_concentrated_at_the_scale_radius_wall(profile_class, scale_radius):
    """Regression, on the grid the published spheroid tabulations were computed on.

    The vertical extent is set by the largest scale in the model, so for any
    spheroid bigger than the disk scale height the grid runs to +-10 r_s in 100
    steps of 0.2 r_s -- which places a wall exactly at z = -r_s. Combined with
    the wall at the axis, that is precisely the corner whose sign the original
    got wrong, and there it put more than half of all the emission into the two
    tiny cells either side of that wall.
    """
    scale_vertical = max(SCALE_HEIGHT, scale_radius)
    radial = np.hstack(
        [
            0.0,
            np.logspace(
                np.log10(1.0e-2 * max(SCALE_RADIAL, scale_radius)),
                np.log10(CUT_OFF * max(SCALE_RADIAL, scale_radius)),
                100,
            ),
        ]
    )
    vertical = np.linspace(-CUT_OFF * scale_vertical, CUT_OFF * scale_vertical, 101)
    assert np.any(np.isclose(vertical, -scale_radius, rtol=0.0, atol=1.0e-15)), (
        "this grid is meant to place a wall at z = -r_s"
    )
    walls = np.broadcast_arrays(radial[:-1, None], radial[1:, None], vertical[None, :-1], vertical[None, 1:])
    mass = profile_class(scale_radius).cell_mass(*walls)
    probability = mass / mass.sum()

    # The profile is symmetric about the midplane, so the emission map must be
    # too. The sign defect broke exactly that, and only below the midplane: it
    # put over half of all the emission into the two cells either side of the
    # z = -r_s wall on the axis, whose mirror images kept their correct,
    # negligible share.
    np.testing.assert_allclose(probability, probability[:, ::-1], rtol=1.0e-6, atol=1.0e-11)
    assert probability[:, : probability.shape[1] // 2].sum() == pytest.approx(0.5, abs=1.0e-9)

    wall = int(np.argmin(np.abs(vertical + scale_radius)))
    flanking = probability[0, wall - 1] + probability[0, wall]
    mirror = probability[0, -wall - 1] + probability[0, -wall]
    assert flanking == pytest.approx(mirror, rel=1.0e-6)
    assert flanking < 0.01


@pytest.mark.parametrize("profile", [HernquistSpheroid(SCALE_RADIAL), JaffeSpheroid(SCALE_RADIAL)])
def test_the_published_grid_barely_needs_the_quadrature_fallback(profile):
    """The closed form covers all but a handful of the published grid outright.

    The radial walls include one at exactly the scale radius and the vertical
    walls one at the midplane, so the four cells meeting at that corner sit on
    the scale sphere and are integrated numerically. Everything else -- 9,996 of
    10,000 cells -- uses the closed form. The whole grid takes a few
    milliseconds either way, so the safety net costs nothing where it matters.
    """
    walls = cell_walls()
    scaled = tuple(wall / profile.scale_radial for wall in walls)
    flagged = profile._ill_conditioned(*scaled)
    assert flagged.sum() <= 4
    assert np.all(np.isfinite(profile.cell_mass(*walls)))
