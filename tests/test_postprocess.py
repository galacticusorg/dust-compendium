"""Tests for turning solved models into a tabulation."""

import numpy as np
import pytest

from dustcompendium.postprocess import (
    EXTRAPOLATION_DECADES,
    attenuation_of,
    fit_extrapolation,
)


class TestAttenuation:
    def test_a_model_divided_by_itself_is_transparent(self):
        values = np.array([[2.0, 3.0]])
        attenuation, _ = attenuation_of(values, values * 0.01, values, values * 0.01)
        np.testing.assert_allclose(attenuation, 1.0)

    def test_attenuation_is_the_ratio(self):
        attenuation, _ = attenuation_of(np.array([1.0]), np.array([0.0]), np.array([4.0]), np.array([0.0]))
        assert attenuation[0] == pytest.approx(0.25)

    def test_both_uncertainties_are_propagated(self):
        """The original divided only the numerator's, understating the result."""
        values, reference = np.array([1.0]), np.array([2.0])
        one_sided = 0.1 / 2.0
        _, uncertainty = attenuation_of(values, np.array([0.1]), reference, np.array([0.2]))
        assert uncertainty[0] > one_sided
        expected = 0.5 * np.sqrt((0.1 / 1.0) ** 2 + (0.2 / 2.0) ** 2)
        assert uncertainty[0] == pytest.approx(expected)

    def test_a_zero_reference_gives_no_answer_rather_than_an_error(self):
        attenuation, _ = attenuation_of(np.array([1.0]), np.array([0.0]), np.array([0.0]), np.array([0.0]))
        assert np.isnan(attenuation[0])


class TestExtrapolation:
    def perfect(self, constant=-0.5, gradient=-0.3):
        # Several points per decade, as the published sixty-point grid has. With
        # only two in the fitting range a straight line passes through both and
        # the residual is identically zero, which says nothing about the form.
        depths = np.hstack([0.0, np.logspace(-2.0, 3.0, 21)])
        with np.errstate(divide="ignore"):
            attenuation = np.exp(constant) * np.where(depths > 0, depths, 1.0) ** gradient
        return depths, attenuation[:, None, None] * np.ones((1, 2, 3))

    def test_a_power_law_is_recovered_exactly(self):
        depths, attenuation = self.perfect()
        coefficients, residual = fit_extrapolation(depths, attenuation)
        np.testing.assert_allclose(coefficients[0], -0.5, rtol=1.0e-9)
        np.testing.assert_allclose(coefficients[1], -0.3, rtol=1.0e-9)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)

    def test_the_coefficients_reproduce_the_transmission(self):
        """They are used as exp(c0 + c1 ln tau), so check that directly."""
        depths, attenuation = self.perfect()
        coefficients, _ = fit_extrapolation(depths, attenuation)
        predicted = np.exp(coefficients[0] + coefficients[1] * np.log(1.0e4))
        assert predicted[0, 0] == pytest.approx(np.exp(-0.5) * 1.0e4**-0.3, rel=1.0e-9)

    def test_departure_from_a_power_law_shows_in_the_residual(self):
        """The point of keeping it: the form is assumed, not derived."""
        depths, attenuation = self.perfect()
        bent = attenuation.copy()
        bent[-1] *= 2.0
        _, residual = fit_extrapolation(depths, bent)
        assert np.all(residual > 0.05)

    def test_only_the_top_decade_is_fitted(self):
        """Changing a depth outside the fitting range must not move the fit."""
        depths, attenuation = self.perfect()
        coefficients, _ = fit_extrapolation(depths, attenuation)
        disturbed = attenuation.copy()
        disturbed[1] *= 10.0
        moved, _ = fit_extrapolation(depths, disturbed)
        np.testing.assert_allclose(coefficients, moved, rtol=1.0e-12)

    def test_two_points_in_range_make_the_residual_meaningless(self):
        """Recorded so a coarse grid's zero residual is not read as a good fit."""
        depths = np.array([0.0, 1.0, 100.0, 1000.0])
        attenuation = np.array([1.0, 0.5, 0.2, 0.02])[:, None] * np.ones((1, 2))
        _, residual = fit_extrapolation(depths, attenuation)
        np.testing.assert_allclose(residual, 0.0, atol=1.0e-12)

    def test_the_zero_entry_is_ignored(self):
        """It cannot take part in a fit in the log."""
        depths, attenuation = self.perfect()
        assert depths[0] == 0.0
        coefficients, _ = fit_extrapolation(depths, attenuation)
        assert np.all(np.isfinite(coefficients))

    def test_the_fitting_range_is_the_documented_factor(self):
        depths = np.array([1.0, 5.0, 20.0, 100.0])
        attenuation = np.ones((4, 1))
        # Only depths within a factor of ten of 100 are used, so 1.0 is excluded.
        coefficients, _ = fit_extrapolation(depths, attenuation, decades=EXTRAPOLATION_DECADES)
        np.testing.assert_allclose(coefficients[1], 0.0, atol=1.0e-12)

    def test_too_few_points_is_refused(self):
        with pytest.raises(ValueError, match="needs at least two"):
            fit_extrapolation(np.array([0.0, 1.0]), np.ones((2, 1)))

    def test_a_mismatched_axis_is_refused(self):
        with pytest.raises(ValueError, match="optical depths against an array"):
            fit_extrapolation(np.array([1.0, 10.0, 100.0]), np.ones((2, 1)))

    def test_a_cell_which_transmits_nothing_is_recorded_as_opaque(self):
        """Not as not-a-number, which the consumer would propagate.

        Galacticus extrapolates beyond the tabulated depths by default and does
        not test for NaN, so a cell left unfitted would silently become a NaN
        attenuation. Recording a floor says the same thing safely.
        """
        from dustcompendium.postprocess import OPAQUE_LOG_TRANSMISSION

        depths, attenuation = self.perfect()
        attenuation[:, 0, 0] = 0.0
        coefficients, residual = fit_extrapolation(depths, attenuation)
        assert coefficients[0, 0, 0] == OPAQUE_LOG_TRANSMISSION
        assert coefficients[1, 0, 0] == 0.0
        # Which evaluates to zero, for any optical depth beyond the table.
        assert np.exp(coefficients[0, 0, 0] + coefficients[1, 0, 0] * np.log(1.0e8)) < 1.0e-300
        # The residual stays absent: there was no fit, so there is no residual,
        # and that keeps these cells identifiable.
        assert np.isnan(residual[0, 0])
        assert np.all(np.isfinite(coefficients[:, 1, 1]))

    def test_partial_underflow_is_opaque_too(self):
        """Deliberately, and not because it could not be fitted.

        Transmission falls with optical depth, so a cell which has underflowed
        at the top of the fitting range has only its deep tail left. Lines
        through numbers that small are noise -- over the Rv3.1 re-run those fits
        came out with gradients from -2654 to +1236, the positive ones
        extrapolating to infinite transmission.
        """
        from dustcompendium.postprocess import OPAQUE_LOG_TRANSMISSION

        depths, attenuation = self.perfect()
        within = (depths > 0) & (depths * 10.0 >= depths[depths > 0].max())
        # Zero only the deepest depth, leaving the rest of the range intact.
        attenuation[np.flatnonzero(within)[-1], 0, 0] = 0.0
        coefficients, residual = fit_extrapolation(depths, attenuation)
        assert coefficients[0, 0, 0] == OPAQUE_LOG_TRANSMISSION
        assert coefficients[1, 0, 0] == 0.0
        assert np.isnan(residual[0, 0])

    def test_no_coefficient_is_ever_not_a_number(self):
        """The property the consumer depends on, over an awkward input."""
        depths, attenuation = self.perfect()
        attenuation[:, 0, 0] = 0.0
        attenuation[-1, 0, 1] = 0.0
        attenuation[-3:, 1, 2] = 0.0
        coefficients, _ = fit_extrapolation(depths, attenuation)
        assert np.isfinite(coefficients).all()
        # And no cell extrapolates to more light than it started with.
        for depth in (1.0e4, 1.0e8):
            assert np.all(np.exp(coefficients[0] + coefficients[1] * np.log(depth)) <= 1.0)

    def test_a_single_positive_depth_is_opaque_rather_than_a_fit(self):
        """One point cannot determine a line, so do not pretend it does."""
        from dustcompendium.postprocess import OPAQUE_LOG_TRANSMISSION

        depths, attenuation = self.perfect()
        within = (depths > 0) & (depths * 10.0 >= depths[depths > 0].max())
        attenuation[np.flatnonzero(within)[:-1], 0, 0] = 0.0
        coefficients, _ = fit_extrapolation(depths, attenuation)
        assert coefficients[0, 0, 0] == OPAQUE_LOG_TRANSMISSION
        assert coefficients[1, 0, 0] == 0.0


@pytest.mark.hyperion
@pytest.mark.solver
@pytest.mark.slow
class TestCollect:
    """End to end: build a small campaign, solve it, and assemble it."""

    CONFIG = """
label: collected
dust: {ferrara: milkyWay, referenceOpacity: 250.0}
geometry:
  cutOff: 10.0
  radialCells: 16
  verticalCells: 16
  spacing: nested
  sampling: average
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137,
             opticalDepth: {minimum: 1.0, maximum: 100.0, count: 3, includeZero: true}}
tabulation:
  wavelengths: {minimum: 0.44, maximum: 0.55, count: 2}
  inclinations: {minimum: 0.0, maximum: 90.0, count: 2, spacing: linear}
  photons: 4000
"""

    @pytest.fixture(scope="class")
    def tabulation(self, tmp_path_factory):
        import subprocess

        from dustcompendium.campaign import Campaign
        from dustcompendium.config import CampaignConfig
        from dustcompendium.dust import ferrara, opacity_to_extinction
        from dustcompendium.hyperion_model import write_model
        from dustcompendium.postprocess import collect
        from dustcompendium.runner import solver_command

        root = tmp_path_factory.mktemp("collected")
        campaign = Campaign(CampaignConfig.from_yaml(self.CONFIG))
        dust = ferrara.build("milkyWay", 250.0)
        for run in campaign.runs():
            source = root / f"{run.file_stem}.hdf5"
            write_model(run.spec, dust, str(source), sampling="average", **campaign.grid_options())
            completed = subprocess.run(
                solver_command(source, root / f"{run.file_stem}.rtout"),
                capture_output=True,
                text=True,
                check=False,
            )
            assert completed.returncode == 0
        return collect(campaign, root, opacity_to_extinction(dust))

    def test_the_unattenuated_model_transmits_everything(self, tabulation):
        """Dividing the dust-free model by itself must give exactly one."""
        np.testing.assert_allclose(tabulation.attenuation["disk"][:, :, 0], 1.0)

    def test_transmission_falls_as_optical_depth_rises(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(np.diff(attenuation, axis=2) < 0.0)

    def test_transmission_stays_between_zero_and_one(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation > 0.0) and np.all(attenuation <= 1.0 + 1.0e-9)

    def test_the_blue_is_attenuated_more_than_the_visual(self, tabulation):
        """Reddening, from the shape of the extinction curve."""
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation[0, :, 1:] < attenuation[1, :, 1:])

    def test_edge_on_is_attenuated_more_than_face_on(self, tabulation):
        attenuation = tabulation.attenuation["disk"]
        assert np.all(attenuation[:, 1, 1:] < attenuation[:, 0, 1:])

    def test_the_shapes_are_what_the_axes_imply(self, tabulation):
        tabulation.check()
        assert tabulation.expected_shape("disk") == (2, 2, 4)

    def test_it_is_a_version_one_tabulation_without_a_spheroid(self, tabulation):
        """One emitter and one axis is not the published layout."""
        assert tabulation.format_version == 2


class TestInsufficientDepths:
    """A campaign can legitimately have too few optical depths to extrapolate."""

    def test_a_single_zero_depth_is_reported_not_crashed_on(self):
        """Regression: taking the largest of an empty selection raised a numpy error.

        A diagnostic campaign at a single optical depth of zero has nothing
        positive to fit against. The guard that catches too few depths asked for
        the largest of them first, which is itself an error when there are none,
        so the clear message was never reached.
        """
        from dustcompendium.postprocess import InsufficientDepths

        with pytest.raises(InsufficientDepths, match="only 0 optical depths"):
            fit_extrapolation(np.array([0.0]), np.ones((1, 2)))

    def test_it_is_a_value_error_so_existing_handling_still_works(self):
        from dustcompendium.postprocess import InsufficientDepths

        assert issubclass(InsufficientDepths, ValueError)

    def test_one_positive_depth_is_also_too_few(self):
        from dustcompendium.postprocess import InsufficientDepths

        with pytest.raises(InsufficientDepths, match="needs at least two"):
            fit_extrapolation(np.array([0.0, 5.0]), np.ones((2, 2)))

    def test_collect_warns_and_fills_rather_than_refusing(self, tmp_path):
        """So a diagnostic campaign can still be assembled and looked at."""
        import warnings

        from dustcompendium.campaign import Campaign
        from dustcompendium.config import CampaignConfig
        from dustcompendium.postprocess import collect

        text = """
label: single
dust: {ferrara: milkyWay}
geometry:
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137, opticalDepth: [0.0]}
tabulation:
  wavelengths: [0.55]
  inclinations: [90.0]
"""
        campaign = Campaign(CampaignConfig.from_yaml(text))
        # Stand in for solved models, so this needs no Hyperion.
        import dustcompendium.postprocess as postprocess

        def fake(path):
            return (np.array([0.55]), np.ones((1, 1)), np.full((1, 1), 0.01))

        original, postprocess.read_sed = postprocess.read_sed, fake
        try:
            for run in campaign.runs():
                (tmp_path / f"{run.file_stem}.rtout").touch()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = collect(campaign, tmp_path, opacity=1.0)
            assert any("no extrapolation" in str(w.message) for w in caught)
            assert np.all(np.isnan(result.extrapolation["disk"]))
        finally:
            postprocess.read_sed = original
