"""Tests for looking profiles up by the names the configurations use."""

import pytest

from dustcompendium.geometry import (
    SPHEROID_PROFILES,
    VERTICAL_STRUCTURES,
    ExponentialVertical,
    HernquistSpheroid,
    JaffeSpheroid,
    SechSquaredVertical,
    spheroid_profile,
    vertical_structure,
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [("exponential", ExponentialVertical), ("sechSquared", SechSquaredVertical)],
)
def test_vertical_structures_resolve(name, expected):
    """The names are those ``runModels.pl`` used for ``diskStructureVertical``."""
    structure = vertical_structure(name, 0.137)
    assert isinstance(structure, expected)
    assert structure.scale_height == pytest.approx(0.137)


@pytest.mark.parametrize(("name", "expected"), [("hernquist", HernquistSpheroid), ("jaffe", JaffeSpheroid)])
def test_spheroid_profiles_resolve(name, expected):
    """The names are those ``runModels.pl`` used for ``spheroidStructure``."""
    profile = spheroid_profile(name, 2.5)
    assert isinstance(profile, expected)
    assert profile.scale_radial == pytest.approx(2.5)


def test_unknown_vertical_structure_lists_the_known_ones():
    with pytest.raises(KeyError, match="exponential, sechSquared"):
        vertical_structure("sech", 1.0)


def test_unknown_spheroid_profile_lists_the_known_ones():
    with pytest.raises(KeyError, match="hernquist, jaffe"):
        spheroid_profile("nfw", 1.0)


def test_registries_and_factories_agree():
    assert set(VERTICAL_STRUCTURES) == {"exponential", "sechSquared"}
    assert set(SPHEROID_PROFILES) == {"hernquist", "jaffe"}


def test_repr_round_trips_the_parameters():
    assert "0.137" in repr(vertical_structure("sechSquared", 0.137))
    assert "2.5" in repr(spheroid_profile("hernquist", 2.5))
