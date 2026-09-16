"""Tests for comparing against a published atlas.

The atlas itself is not vendored -- it is published data, and far too large --
so anything needing it is skipped when it is absent. See ``conftest.py`` for
where it is looked for.
"""

import numpy as np
import pytest

from dustcompendium.validation import (
    ANGSTROMS_PER_MICRON,
    SCALE_RADIUS_PER_EFFECTIVE_RADIUS,
    Comparison,
    compare_with_atlas,
    read_atlas,
)

pytestmark = pytest.mark.reference


class TestReadAtlas:
    def test_wavelengths_come_back_in_microns(self, reference_atlas):
        """The atlas publishes them in Angstroms."""
        atlas = read_atlas(reference_atlas)
        assert atlas.wavelengths.min() == pytest.approx(1250.0 / ANGSTROMS_PER_MICRON)
        assert atlas.wavelengths.max() == pytest.approx(21578.0 / ANGSTROMS_PER_MICRON)

    def test_spheroid_sizes_come_back_as_scale_radii(self, reference_atlas):
        """The atlas tabulates against the effective radius, which is smaller."""
        atlas = read_atlas(reference_atlas)
        np.testing.assert_allclose(
            atlas.scale_radii,
            np.array([0.1, 0.4, 1.6]) * SCALE_RADIUS_PER_EFFECTIVE_RADIUS,
            rtol=1.0e-12,
        )

    def test_the_axes_have_the_shape_the_arrays_imply(self, reference_atlas):
        atlas = read_atlas(reference_atlas)
        assert atlas.attenuation["disk"].shape == (
            atlas.wavelengths.size,
            atlas.inclinations.size,
            atlas.optical_depths.size,
        )
        assert atlas.attenuation["spheroid"].shape == (
            atlas.wavelengths.size,
            atlas.inclinations.size,
            atlas.optical_depths.size,
            atlas.scale_radii.size,
        )

    def test_it_is_the_jaffe_spheroid_ferrara_used(self, reference_atlas):
        assert read_atlas(reference_atlas).metadata["spheroidProfile"] == "Jaffe"

    def test_the_atlas_is_quantized(self, reference_atlas):
        """Published to two decimals, which is the floor on any comparison.

        Some entries exceed one, which a transmission cannot, by exactly the
        rounding: this is what sets the tolerance everything else is judged by.
        """
        values = read_atlas(reference_atlas).attenuation["disk"]
        np.testing.assert_allclose(values, np.round(values, 2), atol=1.0e-12)
        assert values.max() > 1.0


class TestComparison:
    def test_a_difference_is_reported_both_ways(self):
        computed = np.array([0.5, 0.6])
        published = np.array([0.52, 0.58])
        comparison = Comparison("disk", computed, published)
        assert comparison.worst == pytest.approx(0.02)
        assert comparison.median == pytest.approx(0.02)
        assert comparison.fraction_within(0.03) == 1.0
        assert comparison.fraction_within(0.01) == 0.0
        assert "disk" in comparison.summary()

    def test_an_axis_the_tabulation_lacks_is_reported(self, reference_atlas):
        from tests.test_plots import synthetic

        atlas = read_atlas(reference_atlas)
        with pytest.raises(ValueError, match="no entry at"):
            compare_with_atlas(synthetic(), atlas, "disk")
