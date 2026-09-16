"""Tests for truncating a spheroid at a finite radius.

This is the original's ``spheroidCutOff``, documented as "distance (in units of
scale lengths) at which to truncate the spheroid", parsed at
``hyperionBuildModel.py`` line 57 and then never used. The Ferrara-matched
models set it to 5.8 to match a truncation at five effective radii, and did not
get it.
"""

from itertools import pairwise

import numpy as np
import pytest

from dustcompendium.geometry import HernquistSpheroid, JaffeSpheroid, spheroid_profile

TRUNCATION = 5.8

#: Mass enclosed within ``t`` scale radii, per unit normalization and scale
#: radius cubed, integrated by hand from the profiles.
ENCLOSED = {
    HernquistSpheroid: lambda t: 4.0 * np.pi * (0.5 - 1.0 / (1.0 + t) + 0.5 / (1.0 + t) ** 2),
    JaffeSpheroid: lambda t: 4.0 * np.pi * (1.0 - 1.0 / (1.0 + t)),
}

PROFILES = [pytest.param(cls, id=cls.__name__) for cls in ENCLOSED]


@pytest.mark.parametrize("profile_class", PROFILES)
class TestTruncation:
    def test_density_vanishes_beyond_the_truncation(self, profile_class):
        profile = profile_class(1.0, truncation=TRUNCATION)
        assert float(profile.density(TRUNCATION - 0.01, 0.0)) > 0.0
        assert float(profile.density(TRUNCATION + 0.01, 0.0)) == 0.0
        assert float(profile.density(0.0, TRUNCATION + 0.01)) == 0.0

    def test_the_truncation_is_spherical_not_cylindrical(self, profile_class):
        """A point beyond the radius diagonally is outside, as for any sphere."""
        profile = profile_class(1.0, truncation=2.0)
        assert float(profile.density(1.9, 0.0)) > 0.0
        assert float(profile.density(1.5, 1.5)) == 0.0

    @pytest.mark.parametrize("truncation", [1.5, 5.8, 10.0, 100.0])
    def test_total_mass_matches_the_enclosed_mass(self, profile_class, truncation):
        profile = profile_class(1.0, truncation=truncation)
        reach = 2.0 * truncation
        total = float(profile.cell_mass(0.0, reach, -reach, reach))
        assert total == pytest.approx(ENCLOSED[profile_class](truncation), rel=1.0e-9)

    def test_mass_is_unchanged_well_inside_the_truncation(self, profile_class):
        """Cells the truncation does not reach keep the closed form untouched."""
        cell = (0.3, 0.7, 0.4, 0.9)
        assert float(profile_class(1.0, truncation=TRUNCATION).cell_mass(*cell)) == pytest.approx(
            float(profile_class(1.0).cell_mass(*cell)), rel=1.0e-12
        )

    def test_cells_beyond_the_truncation_are_empty(self, profile_class):
        profile = profile_class(1.0, truncation=2.0)
        assert float(profile.cell_mass(3.0, 4.0, 0.5, 1.0)) == 0.0
        assert float(profile.cell_mass(0.0, 0.5, 5.0, 6.0)) == 0.0

    def test_mass_is_additive_across_the_truncation(self, profile_class):
        """Splitting a cell the truncation crosses must not change its mass."""
        profile = profile_class(1.0, truncation=TRUNCATION)
        whole = float(profile.cell_mass(0.5, 9.0, 0.2, 3.0))
        walls = [0.5, 3.0, 5.7, 5.9, 9.0]
        parts = sum(float(profile.cell_mass(lower, upper, 0.2, 3.0)) for lower, upper in pairwise(walls))
        assert parts == pytest.approx(whole, rel=1.0e-9)

    def test_truncation_lowers_the_optical_depth_column(self, profile_class):
        """The ray at the scale radius now leaves the sphere, so the column is shorter."""
        truncated = profile_class(1.0, truncation=TRUNCATION)
        full = profile_class(1.0)
        assert truncated.optical_depth_integral < full.optical_depth_integral
        assert truncated.optical_depth_integral == pytest.approx(full.optical_depth_integral, rel=0.02)

    def test_normalization_still_reproduces_the_requested_optical_depth(self, profile_class):
        """The whole point of the column: truncated or not, tau comes back."""
        from scipy.integrate import quad

        profile = profile_class(1.0, truncation=TRUNCATION)
        optical_depth, opacity = 2.3, 0.7
        normalization = profile.density_normalization(optical_depth, opacity)
        column, _ = quad(
            lambda z: normalization * float(profile.density(profile.optical_depth_radius, z)),
            -np.inf,
            np.inf,
            limit=200,
        )
        assert opacity * column == pytest.approx(optical_depth, rel=1.0e-6)

    def test_a_far_truncation_approaches_the_untruncated_profile(self, profile_class):
        far = profile_class(1.0, truncation=1.0e6)
        full = profile_class(1.0)
        assert far.optical_depth_integral == pytest.approx(full.optical_depth_integral, rel=1.0e-6)

    def test_truncation_radius_is_reported_in_length_units(self, profile_class):
        assert profile_class(3.0, truncation=2.0).truncation_radius == pytest.approx(6.0)
        assert profile_class(3.0).truncation_radius == np.inf

    @pytest.mark.parametrize("truncation", [1.0, 0.5, -1.0])
    def test_a_truncation_inside_the_scale_radius_is_rejected(self, profile_class, truncation):
        """The optical depth ray runs at the scale radius; inside it there is nothing."""
        with pytest.raises(ValueError, match="must exceed one scale radius"):
            profile_class(1.0, truncation=truncation)

    def test_repr_shows_the_truncation(self, profile_class):
        assert "truncation" in repr(profile_class(1.0, truncation=TRUNCATION))
        assert "truncation" not in repr(profile_class(1.0))


def test_the_registry_passes_the_truncation_through():
    profile = spheroid_profile("hernquist", 2.0, truncation=5.8)
    assert isinstance(profile, HernquistSpheroid)
    assert profile.truncation == pytest.approx(5.8)
    assert spheroid_profile("jaffe", 2.0).truncation is None
