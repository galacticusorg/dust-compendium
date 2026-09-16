"""Tests for the cylindrical grid."""

import numpy as np
import pytest

from dustcompendium.galaxy import Component, Galaxy
from dustcompendium.geometry import ExponentialDisk, HernquistSpheroid, SechSquaredVertical
from dustcompendium.grid import CylindricalGrid

CUT_OFF = 10.0


def published_grid():
    galaxy = Galaxy(
        [
            Component(
                "disk",
                stellar=ExponentialDisk(1.0, SechSquaredVertical(0.137)),
                dust=ExponentialDisk(1.0, SechSquaredVertical(0.137)),
            )
        ]
    )
    return CylindricalGrid.for_galaxy(galaxy, CUT_OFF)


class TestConstruction:
    def test_matches_the_published_grid(self):
        """The walls of ``hyperionBuildModel.py`` lines 101-105."""
        grid = published_grid()
        radial = np.hstack([0.0, np.logspace(np.log10(1.0e-2), np.log10(CUT_OFF), 100)])
        vertical = np.linspace(-CUT_OFF * 0.137, CUT_OFF * 0.137, 101)
        np.testing.assert_allclose(grid.radial_walls, radial, rtol=1.0e-14)
        np.testing.assert_allclose(grid.vertical_walls, vertical, rtol=1.0e-14)
        assert grid.shape == (1, 100, 100)

    def test_the_axis_is_a_wall(self):
        assert published_grid().radial_walls[0] == 0.0

    def test_an_even_cell_count_puts_a_wall_on_the_midplane(self):
        assert published_grid().midplane_is_a_wall

    def test_an_odd_cell_count_straddles_the_midplane(self):
        """Recorded because a straddling cell is what the original got wrong."""
        galaxy = Galaxy([Component("disk", stellar=ExponentialDisk(1.0, SechSquaredVertical(0.137)))])
        grid = CylindricalGrid.for_galaxy(galaxy, CUT_OFF, vertical_cells=99)
        assert not grid.midplane_is_a_wall

    def test_a_spheroid_can_set_the_vertical_extent(self):
        galaxy = Galaxy(
            [
                Component("disk", stellar=ExponentialDisk(1.0, SechSquaredVertical(0.137))),
                Component("spheroid", stellar=HernquistSpheroid(2.0)),
            ]
        )
        grid = CylindricalGrid.for_galaxy(galaxy, CUT_OFF)
        assert grid.vertical_walls[-1] == pytest.approx(CUT_OFF * 2.0)
        assert grid.radial_walls[-1] == pytest.approx(CUT_OFF * 2.0)

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cut_off": -1.0}, "cut off must be positive"),
            ({"cut_off": 10.0, "radial_cells": 1}, "at least two radial cells"),
        ],
    )
    def test_invalid_arguments_are_rejected(self, kwargs, match):
        galaxy = Galaxy([Component("disk", stellar=ExponentialDisk(1.0, SechSquaredVertical(0.137)))])
        with pytest.raises(ValueError, match=match):
            CylindricalGrid.for_galaxy(galaxy, **kwargs)

    def test_unordered_walls_are_rejected(self):
        with pytest.raises(ValueError, match="strictly increasing"):
            CylindricalGrid(np.array([0.0, 2.0, 1.0]), np.array([-1.0, 1.0]), np.array([0.0, 6.3]))


class TestDerivedQuantities:
    def test_centres_lie_inside_their_cells(self):
        grid = published_grid()
        assert np.all(grid.radial_centres > grid.radial_walls[:-1])
        assert np.all(grid.radial_centres < grid.radial_walls[1:])
        assert np.all(grid.vertical_centres > grid.vertical_walls[:-1])
        assert np.all(grid.vertical_centres < grid.vertical_walls[1:])

    def test_radial_centres_are_geometric_beyond_the_axis(self):
        grid = published_grid()
        walls = grid.radial_walls
        np.testing.assert_allclose(grid.radial_centres[1:], np.sqrt(walls[1:-1] * walls[2:]), rtol=1.0e-12)

    def test_the_innermost_centre_is_half_the_outer_wall(self):
        """The geometric mean is undefined against the axis, so Hyperion halves."""
        grid = published_grid()
        assert grid.radial_centres[0] == pytest.approx(grid.radial_walls[1] / 2.0)

    def test_vertical_centres_are_arithmetic(self):
        grid = published_grid()
        walls = grid.vertical_walls
        np.testing.assert_allclose(grid.vertical_centres, (walls[:-1] + walls[1:]) / 2.0, rtol=1.0e-14)

    def test_centre_and_bound_arrays_have_the_grid_shape(self):
        grid = published_grid()
        radius, height = grid.centres
        assert radius.shape == grid.shape and height.shape == grid.shape
        assert all(bound.shape == grid.shape for bound in grid.bounds)

    def test_volumes_sum_to_the_enclosing_cylinder(self):
        grid = published_grid()
        enclosing = np.pi * grid.radial_walls[-1] ** 2 * (grid.vertical_walls[-1] - grid.vertical_walls[0])
        assert grid.cell_volumes.sum() == pytest.approx(enclosing, rel=1.0e-12)

    def test_cell_masses_can_be_taken_straight_from_the_bounds(self):
        grid = published_grid()
        profile = ExponentialDisk(1.0, SechSquaredVertical(0.137))
        mass = profile.cell_mass(*grid.bounds)
        assert mass.shape == grid.shape
        assert np.all(mass >= 0.0)


@pytest.mark.hyperion
def test_centres_agree_with_hyperion():
    """The dust density is sampled at cell centres, so the convention must match."""
    from hyperion.model import Model

    grid = published_grid()
    model = Model()
    model.set_cylindrical_polar_grid(grid.radial_walls, grid.vertical_walls, grid.azimuthal_walls)
    assert model.grid.shape == grid.shape
    radius, height = grid.centres
    np.testing.assert_allclose(model.grid.gw, radius, rtol=1.0e-14)
    np.testing.assert_allclose(model.grid.gz, height, rtol=1.0e-14)
    np.testing.assert_allclose(model.grid.volumes, grid.cell_volumes, rtol=1.0e-12)


class TestNestedSpacing:
    """A grid that resolves the smallest scale while reaching the largest.

    The published scheme sizes every cell by the largest component, so a thin
    disk inside a large spheroid is left unresolved. Hyperion's cylindrical grid
    takes arbitrary wall positions, so this needs no change of grid type -- its
    AMR grid is Cartesian, and would cost the cylindrical symmetry.
    """

    def galaxy(self, spheroid_scale):
        return Galaxy(
            [
                Component(
                    "disk",
                    stellar=ExponentialDisk(1.0, SechSquaredVertical(0.137)),
                    dust=ExponentialDisk(1.0, SechSquaredVertical(0.137)),
                ),
                Component("spheroid", stellar=HernquistSpheroid(spheroid_scale)),
            ]
        )

    @pytest.mark.parametrize("spheroid_scale", [1.0, 10.0, 100.0])
    def test_the_finest_cell_resolves_the_smallest_scale(self, spheroid_scale):
        galaxy = self.galaxy(spheroid_scale)
        grid = CylindricalGrid.for_galaxy(galaxy, CUT_OFF, spacing="nested")
        assert np.diff(grid.vertical_walls).min() < galaxy.vertical_scales[0]
        assert grid.radial_walls[1] < galaxy.radial_scales[0]

    @pytest.mark.parametrize("spheroid_scale", [1.0, 10.0, 100.0])
    def test_the_grid_still_reaches_the_largest_scale(self, spheroid_scale):
        galaxy = self.galaxy(spheroid_scale)
        grid = CylindricalGrid.for_galaxy(galaxy, CUT_OFF, spacing="nested")
        assert grid.vertical_walls[-1] == pytest.approx(CUT_OFF * galaxy.extent_vertical)
        assert grid.radial_walls[-1] == pytest.approx(CUT_OFF * galaxy.extent_radial)

    def test_the_published_scheme_does_not_resolve_a_large_spheroid(self):
        """The behaviour nesting exists to fix, recorded so it cannot regress."""
        galaxy = self.galaxy(10.0)
        grid = CylindricalGrid.for_galaxy(galaxy, CUT_OFF)
        assert np.diff(grid.vertical_walls).min() > 10.0 * galaxy.vertical_scales[0]

    def test_the_midplane_is_still_a_wall(self):
        grid = CylindricalGrid.for_galaxy(self.galaxy(10.0), CUT_OFF, spacing="nested")
        assert grid.midplane_is_a_wall

    def test_walls_remain_symmetric_and_increasing(self):
        grid = CylindricalGrid.for_galaxy(self.galaxy(10.0), CUT_OFF, spacing="nested")
        np.testing.assert_allclose(grid.vertical_walls, -grid.vertical_walls[::-1], atol=1.0e-12)
        assert np.all(np.diff(grid.vertical_walls) > 0.0)

    def test_the_two_schemes_agree_when_every_scale_matches(self):
        """With one vertical and one radial scale there is nothing to nest."""
        galaxy = Galaxy([Component("disk", stellar=ExponentialDisk(1.0, SechSquaredVertical(1.0)))])
        published = CylindricalGrid.for_galaxy(galaxy, CUT_OFF)
        nested = CylindricalGrid.for_galaxy(galaxy, CUT_OFF, spacing="nested")
        np.testing.assert_allclose(published.vertical_walls, nested.vertical_walls)
        np.testing.assert_allclose(published.radial_walls, nested.radial_walls)

    def test_volumes_still_tile_the_cylinder(self):
        grid = CylindricalGrid.for_galaxy(self.galaxy(10.0), CUT_OFF, spacing="nested")
        enclosing = np.pi * grid.radial_walls[-1] ** 2 * (grid.vertical_walls[-1] - grid.vertical_walls[0])
        assert grid.cell_volumes.sum() == pytest.approx(enclosing, rel=1.0e-12)

    def test_an_unknown_scheme_is_rejected(self):
        with pytest.raises(ValueError, match="spacing must be"):
            CylindricalGrid.for_galaxy(self.galaxy(1.0), CUT_OFF, spacing="adaptive")
