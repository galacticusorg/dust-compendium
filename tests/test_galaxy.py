"""Tests for the component/role galaxy model."""

import pytest

from dustcompendium.galaxy import Component, Galaxy
from dustcompendium.geometry import (
    ExponentialDisk,
    HernquistSpheroid,
    SechSquaredVertical,
)


def disk(scale_radial=1.0, scale_height=0.137):
    return ExponentialDisk(scale_radial, SechSquaredVertical(scale_height))


def published_galaxy():
    """Disk stars and disk dust, with a stellar spheroid: the published geometry."""
    return Galaxy(
        [
            Component("disk", stellar=disk(), dust=disk()),
            Component("spheroid", stellar=HernquistSpheroid(1.0)),
        ]
    )


class TestComponent:
    def test_roles_are_reported(self):
        both = Component("disk", stellar=disk(), dust=disk())
        assert both.emits and both.attenuates

    def test_a_component_may_hold_only_dust(self):
        dust_only = Component("halo", dust=HernquistSpheroid(1.0))
        assert dust_only.attenuates
        assert not dust_only.emits

    def test_a_component_may_hold_only_stars(self):
        stars_only = Component("spheroid", stellar=HernquistSpheroid(1.0))
        assert stars_only.emits
        assert not stars_only.attenuates

    def test_an_unnamed_component_is_rejected(self):
        with pytest.raises(ValueError, match="needs a name"):
            Component("", stellar=disk())


class TestGalaxy:
    def test_components_are_found_by_name(self):
        galaxy = published_galaxy()
        assert galaxy["spheroid"].stellar is not None
        assert galaxy.names == ("disk", "spheroid")
        assert len(galaxy) == 2

    def test_an_unknown_name_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="disk, spheroid"):
            published_galaxy()["bulge"]

    def test_roles_are_collected(self):
        galaxy = published_galaxy()
        assert [component.name for component in galaxy.emitting] == ["disk", "spheroid"]
        assert [component.name for component in galaxy.attenuating] == ["disk"]

    def test_dust_can_be_added_to_the_spheroid_without_structural_change(self):
        """The whole point of the component model: this is the upcoming work."""
        galaxy = Galaxy(
            [
                Component("disk", stellar=disk(), dust=disk()),
                Component(
                    "spheroid",
                    stellar=HernquistSpheroid(1.0),
                    dust=HernquistSpheroid(1.0),
                ),
            ]
        )
        assert [component.name for component in galaxy.attenuating] == ["disk", "spheroid"]

    def test_extents_take_the_largest_scale(self):
        galaxy = Galaxy(
            [
                Component("disk", stellar=disk(scale_radial=1.0, scale_height=0.137)),
                Component("spheroid", stellar=HernquistSpheroid(3.0)),
            ]
        )
        assert galaxy.extent_radial == pytest.approx(3.0)
        assert galaxy.extent_vertical == pytest.approx(3.0)

    def test_a_small_spheroid_leaves_the_disk_setting_the_vertical_extent(self):
        galaxy = Galaxy(
            [
                Component("disk", stellar=disk(scale_radial=1.0, scale_height=0.137)),
                Component("spheroid", stellar=HernquistSpheroid(0.05)),
            ]
        )
        assert galaxy.extent_radial == pytest.approx(1.0)
        assert galaxy.extent_vertical == pytest.approx(0.137)

    def test_dust_scale_height_counts_towards_the_extent(self):
        """The original took the maximum over stellar and dust scales alike."""
        galaxy = Galaxy([Component("disk", stellar=disk(scale_height=0.1), dust=disk(scale_height=0.9))])
        assert galaxy.extent_vertical == pytest.approx(0.9)

    def test_duplicate_names_are_rejected(self):
        with pytest.raises(ValueError, match="repeated: disk"):
            Galaxy([Component("disk", stellar=disk()), Component("disk", dust=disk())])

    def test_an_empty_galaxy_is_rejected(self):
        with pytest.raises(ValueError, match="at least one component"):
            Galaxy([])
