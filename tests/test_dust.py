"""Tests for the grain optical properties."""

import numpy as np
import pytest

from dustcompendium.dust import SPEED_OF_LIGHT_ANGSTROMS, V_BAND_WAVELENGTH, ferrara

# The literals carried by hyperionDustFerrara.py, in the order it wrote them
# (ascending wavelength), before its flipud.
LEGACY_WAVELENGTH = [
    1250,
    1515,
    1775,
    1995,
    2215,
    2480,
    2895,
    3605,
    4413,
    5512,
    6594,
    8059,
    12369,
    16464,
    21578,
]
LEGACY = {
    "MW": {
        "albedo": [0.60, 0.67, 0.65, 0.55, 0.46, 0.56, 0.61, 0.63, 0.61, 0.59, 0.57, 0.55, 0.53, 0.51, 0.50],
        "asymmetry": [
            0.75,
            0.75,
            0.73,
            0.72,
            0.71,
            0.70,
            0.69,
            0.65,
            0.63,
            0.61,
            0.57,
            0.53,
            0.47,
            0.45,
            0.43,
        ],
        "extinction": [
            3.11,
            2.63,
            2.50,
            2.78,
            3.12,
            2.35,
            2.00,
            1.52,
            1.32,
            1.00,
            0.76,
            0.48,
            0.28,
            0.167,
            0.095,
        ],
    },
    "SMC": {
        "albedo": [0.40, 0.40, 0.58, 0.58, 0.55, 0.56, 0.53, 0.46, 0.43, 0.43, 0.41, 0.38, 0.33, 0.30, 0.29],
        "asymmetry": [
            0.53,
            0.53,
            0.54,
            0.51,
            0.46,
            0.37,
            0.35,
            0.34,
            0.32,
            0.29,
            0.26,
            0.23,
            0.21,
            0.23,
            0.22,
        ],
        "extinction": [
            5.00,
            4.36,
            3.51,
            3.20,
            2.90,
            2.40,
            2.13,
            1.58,
            1.35,
            1.00,
            0.74,
            0.52,
            0.28,
            0.17,
            0.11,
        ],
    },
}


@pytest.mark.parametrize("grain_type", ["MW", "SMC"])
class TestGordonTable:
    def test_matches_the_original_literals(self, grain_type):
        """The data file must carry exactly what the original had inline."""
        table = ferrara.read_table(grain_type)
        order = np.argsort(table.wavelength)
        np.testing.assert_allclose(table.wavelength[order], LEGACY_WAVELENGTH)
        for name, values in LEGACY[grain_type].items():
            np.testing.assert_allclose(getattr(table, name)[order], values, rtol=1.0e-12)

    def test_is_ordered_by_ascending_frequency(self, grain_type):
        """Which is what Hyperion wants, and the reverse of how it is published."""
        table = ferrara.read_table(grain_type)
        assert np.all(np.diff(table.frequency) > 0.0)
        assert np.all(np.diff(table.wavelength) < 0.0)

    def test_extinction_is_relative_to_the_v_band(self, grain_type):
        table = ferrara.read_table(grain_type)
        v_band = table.extinction[table.wavelength == 5512]
        assert v_band == pytest.approx(1.0)

    def test_the_long_and_short_names_agree(self, grain_type):
        long_name = {"MW": "milkyWay", "SMC": "smallMagellanicCloud"}[grain_type]
        np.testing.assert_array_equal(
            ferrara.read_table(grain_type).extinction,
            ferrara.read_table(long_name).extinction,
        )

    def test_frequency_conversion_uses_the_documented_constant(self, grain_type):
        table = ferrara.read_table(grain_type)
        np.testing.assert_allclose(table.frequency, SPEED_OF_LIGHT_ANGSTROMS / table.wavelength, rtol=1.0e-14)


def test_unknown_grain_type_lists_the_known_ones():
    with pytest.raises(KeyError, match="MW"):
        ferrara.read_table("LMC")


@pytest.mark.hyperion
class TestBuiltDust:
    def test_v_band_opacity_follows_the_reference(self):
        """The extinction curve is relative, so the reference sets the scale."""
        from dustcompendium.dust import opacity_to_extinction

        unit = opacity_to_extinction(ferrara.build("milkyWay"))
        scaled = opacity_to_extinction(ferrara.build("milkyWay", reference_opacity=250.0))
        assert unit == pytest.approx(1.0, rel=0.01)
        assert scaled == pytest.approx(250.0 * unit, rel=1.0e-9)

    def test_properties_are_extrapolated_across_the_full_range(self):
        """The tabulation spans 0.125 to 2.16 microns; models go well outside it."""
        from dustcompendium.dust import opacity_to_extinction

        dust = ferrara.build("milkyWay")
        for wavelength in (0.01, 0.1, 1.0, 100.0):
            assert np.isfinite(opacity_to_extinction(dust, wavelength))

    def test_the_two_grain_types_differ(self):
        from dustcompendium.dust import opacity_to_extinction

        milky_way = ferrara.build("milkyWay")
        magellanic = ferrara.build("smallMagellanicCloud")
        ultraviolet = 0.125
        assert opacity_to_extinction(magellanic, ultraviolet) > opacity_to_extinction(milky_way, ultraviolet)

    def test_a_non_positive_reference_is_rejected(self):
        with pytest.raises(ValueError, match="reference opacity must be positive"):
            ferrara.build("milkyWay", reference_opacity=0.0)

    def test_v_band_constant_is_where_it_is_expected(self):
        assert V_BAND_WAVELENGTH == pytest.approx(0.55)


def test_the_tabulation_ships_with_the_package():
    """Regression: `data/` in .gitignore once swept up this package data file.

    It was untracked and absent from the wheel, so the tests passed from the
    source tree while an installed copy would have failed to find it. Reading it
    through importlib.resources is what an installed copy does.
    """
    from importlib import resources

    source = resources.files("dustcompendium.dust").joinpath("data/gordon1997.csv")
    assert source.is_file()
    assert "Gordon" in source.read_text(encoding="utf-8")
