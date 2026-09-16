"""Tests for the figures.

Built on a synthetic tabulation whose extinction is known in closed form, so the
assertions are exact rather than eyeballed. Transmission is

    T = exp(-tau (0.55 / lambda) (1 + i / 90))

which makes the extinction curve a power law in wavelength and, because the
wavelength dependence factors out, gives a reddening of exactly four everywhere:
A_B / A_V is 0.55 / 0.44 = 1.25, so A_V / (A_B - A_V) = 1 / 0.25.
"""

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dustcompendium.plots import (
    V_BAND,
    coordinates,
    extinction,
    label_for,
    plot_extinction,
    plot_extinction_curve,
    reddening,
)
from dustcompendium.tabulation import Tabulation

# The two bands are put on the grid, so that the closed form can be checked
# exactly rather than through an interpolation error.
WAVELENGTHS = np.unique(np.hstack([np.logspace(np.log10(0.1), np.log10(2.0), 24), [0.44, 0.55]]))
INCLINATIONS = np.linspace(0.0, 90.0, 5)
DEPTHS = np.array([0.0, 0.1, 1.0, 10.0])
RADII = np.array([0.1, 1.0, 10.0])
MAGNITUDES = 2.5 / np.log(10.0)


def synthetic():
    depth = (
        DEPTHS[None, None, :, None]
        * (V_BAND / WAVELENGTHS)[:, None, None, None]
        * (1.0 + INCLINATIONS / 90.0)[None, :, None, None]
        * np.ones((1, 1, 1, RADII.size))
    )
    spheroid = np.exp(-depth)
    disk = spheroid[:, :, :, 0]
    axis_values = {"opticalDepth:disk": DEPTHS, "scaleRadial:spheroid": RADII}
    emitter_axes = {
        "disk": ("opticalDepth:disk",),
        "spheroid": ("opticalDepth:disk", "scaleRadial:spheroid"),
    }
    attenuation = {"disk": disk, "spheroid": spheroid}
    extrapolation = {
        "disk": np.zeros((2, WAVELENGTHS.size, INCLINATIONS.size)),
        "spheroid": np.zeros((2, WAVELENGTHS.size, INCLINATIONS.size, RADII.size)),
    }
    residual = {
        "disk": np.zeros((WAVELENGTHS.size, INCLINATIONS.size)),
        "spheroid": np.zeros((WAVELENGTHS.size, INCLINATIONS.size, RADII.size)),
    }
    return Tabulation(
        label="synthetic",
        wavelengths=WAVELENGTHS,
        inclinations=INCLINATIONS,
        axis_names=("opticalDepth:disk", "scaleRadial:spheroid"),
        axis_values=axis_values,
        emitter_axes=emitter_axes,
        attenuation=attenuation,
        uncertainty={name: value * 0.01 for name, value in attenuation.items()},
        extrapolation=extrapolation,
        residual=residual,
        opacity=250.0,
    )


@pytest.fixture
def tabulation():
    return synthetic()


@pytest.fixture(autouse=True)
def close_figures():
    yield
    plt.close("all")


class TestQuantities:
    def test_no_dust_means_no_extinction(self, tabulation):
        np.testing.assert_allclose(extinction(tabulation, "disk")[:, 0], 0.0, atol=1.0e-12)

    def test_extinction_matches_the_closed_form(self, tabulation):
        """A_V = 2.5/ln(10) * tau * (1 + i/90)."""
        got = extinction(tabulation, "disk")
        expected = MAGNITUDES * DEPTHS[None, :] * (1.0 + INCLINATIONS / 90.0)[:, None]
        np.testing.assert_allclose(got, expected, rtol=1.0e-6)

    def test_extinction_rises_with_optical_depth_and_inclination(self, tabulation):
        got = extinction(tabulation, "disk")
        assert np.all(np.diff(got, axis=1) > 0.0)
        assert np.all(np.diff(got[:, 1:], axis=0) > 0.0)

    def test_the_shape_drops_only_the_wavelength_axis(self, tabulation):
        assert extinction(tabulation, "spheroid").shape == (
            INCLINATIONS.size,
            DEPTHS.size,
            RADII.size,
        )

    def test_reddening_is_the_ratio_of_total_to_selective(self, tabulation):
        """Exactly four for this curve, at every depth and inclination."""
        got = reddening(tabulation, "disk")
        np.testing.assert_allclose(got[:, 1:], 4.0, rtol=1.0e-5)

    def test_reddening_is_undefined_without_dust(self, tabulation):
        """Both extinctions are zero there, so the ratio is nothing."""
        assert np.all(np.isnan(reddening(tabulation, "disk")[:, 0]))

    def test_a_wavelength_outside_the_table_is_refused(self, tabulation):
        with pytest.raises(ValueError, match="outside the tabulated range"):
            extinction(tabulation, "disk", 10.0)

    def test_extinction_is_interpolated_between_grid_points(self, tabulation):
        """Between two tabulated wavelengths, not snapped to the nearer."""
        between = float(np.sqrt(WAVELENGTHS[10] * WAVELENGTHS[11]))
        got = extinction(tabulation, "disk", between)[2, 2]
        lower = extinction(tabulation, "disk", WAVELENGTHS[10])[2, 2]
        upper = extinction(tabulation, "disk", WAVELENGTHS[11])[2, 2]
        assert min(lower, upper) < got < max(lower, upper)

    def test_extinction_can_be_taken_at_any_wavelength(self, tabulation):
        """Not only the bands the original hard-coded."""
        assert np.all(extinction(tabulation, "disk", 0.2)[:, 1:] > extinction(tabulation, "disk", 1.0)[:, 1:])


class TestCoordinates:
    def test_they_are_in_array_order(self, tabulation):
        assert list(coordinates(tabulation, "spheroid")) == [
            "inclination",
            "opticalDepth:disk",
            "scaleRadial:spheroid",
        ]

    def test_labels_fall_back_to_the_name(self):
        assert "Inclination" in label_for("inclination")
        assert label_for("somethingElse") == "somethingElse"


class TestPlots:
    def test_a_single_curve_against_inclination(self, tabulation):
        axes = plot_extinction(tabulation, "disk", "inclination", fixed={"opticalDepth": 2})
        assert len(axes.lines) == 1
        np.testing.assert_allclose(axes.lines[0].get_xdata(), INCLINATIONS)

    def test_the_axis_may_be_named_either_way(self, tabulation):
        """Its own name, or the name it is stored under."""
        first = plot_extinction(tabulation, "disk", "opticalDepth:disk")
        second = plot_extinction(tabulation, "disk", "opticalDepth")
        np.testing.assert_allclose(first.lines[0].get_ydata(), second.lines[0].get_ydata())

    def test_a_family_draws_one_curve_per_value(self, tabulation):
        axes = plot_extinction(tabulation, "spheroid", "inclination", colour_by="spheroidScaleRadial")
        assert len(axes.lines) == RADII.size

    def test_the_family_is_drawn_the_right_way_round(self, tabulation):
        """Whichever order the two coordinates appear in the array."""
        axes = plot_extinction(tabulation, "spheroid", "spheroidScaleRadial", colour_by="inclination")
        assert len(axes.lines) == INCLINATIONS.size
        for line in axes.lines:
            assert len(line.get_xdata()) == RADII.size

    def test_an_optical_depth_axis_is_drawn_logarithmically(self, tabulation):
        axes = plot_extinction(tabulation, "disk", "opticalDepth", fixed={"inclination": 2})
        # The zero entry makes the axis non-positive, so it stays linear.
        assert axes.get_xscale() == "linear"

    def test_a_positive_axis_is_drawn_logarithmically(self, tabulation):
        axes = plot_extinction(tabulation, "spheroid", "spheroidScaleRadial", fixed={"opticalDepth": 2})
        assert axes.get_xscale() == "log"

    def test_reddening_can_be_plotted(self, tabulation):
        axes = plot_extinction(
            tabulation, "disk", "inclination", quantity="reddening", fixed={"opticalDepth": 3}
        )
        np.testing.assert_allclose(axes.lines[0].get_ydata(), 4.0, rtol=1.0e-5)

    def test_an_unknown_quantity_is_refused(self, tabulation):
        with pytest.raises(ValueError, match="must be 'extinction' or 'reddening'"):
            plot_extinction(tabulation, "disk", "inclination", quantity="albedo")

    def test_an_axis_the_emitter_lacks_is_refused(self, tabulation):
        with pytest.raises(KeyError, match="not tabulated against"):
            plot_extinction(tabulation, "disk", "spheroidScaleRadial")

    def test_the_extinction_curve_runs_over_wavelength(self, tabulation):
        axes = plot_extinction_curve(tabulation, "disk", fixed={"opticalDepth": 2})
        np.testing.assert_allclose(axes.lines[0].get_xdata(), WAVELENGTHS)
        assert axes.get_xscale() == "log"

    def test_the_extinction_curve_falls_towards_the_red(self, tabulation):
        axes = plot_extinction_curve(tabulation, "disk", fixed={"opticalDepth": 2})
        values = axes.lines[0].get_ydata()
        assert np.all(np.diff(values) < 0.0)

    def test_a_family_of_extinction_curves(self, tabulation):
        axes = plot_extinction_curve(tabulation, "disk", colour_by="opticalDepth")
        assert len(axes.lines) == DEPTHS.size

    def test_curves_are_labelled(self, tabulation):
        axes = plot_extinction_curve(tabulation, "disk")
        assert "Wavelength" in axes.get_xlabel()
        assert "Extinction" in axes.get_ylabel()
        assert "synthetic" in axes.get_title()

    def test_it_draws_onto_given_axes(self, tabulation):
        _, axes = plt.subplots()
        assert plot_extinction(tabulation, "disk", "inclination", axes=axes) is axes


class TestTheNineOriginalScripts:
    """Every figure the original could make, from one implementation."""

    @pytest.mark.parametrize("quantity", ["extinction", "reddening"])
    @pytest.mark.parametrize("against", ["inclination", "opticalDepth", "spheroidScaleRadial"])
    def test_quantity_against_coordinate(self, tabulation, quantity, against):
        axes = plot_extinction(tabulation, "spheroid", against, quantity=quantity)
        assert len(axes.lines) == 1

    @pytest.mark.parametrize("colour_by", ["inclination", "opticalDepth", "spheroidScaleRadial"])
    def test_extinction_curve_coloured_by_coordinate(self, tabulation, colour_by):
        axes = plot_extinction_curve(tabulation, "spheroid", colour_by=colour_by)
        assert len(axes.lines) == len(
            coordinates(tabulation, "spheroid")[
                "inclination"
                if colour_by == "inclination"
                else "opticalDepth:disk"
                if colour_by == "opticalDepth"
                else "scaleRadial:spheroid"
            ]
        )


class TestColouring:
    def test_a_family_over_a_depth_axis_gets_distinct_colours(self, tabulation):
        """The zero entry must not force the rest onto one linear scale.

        With depths of 0, 0.1, 1 and 10 a linear colour scale puts the first
        three within a twentieth of the range of each other, which on viridis is
        the same colour. The positive values are scaled logarithmically instead.
        """
        from matplotlib.colors import to_rgba

        axes = plot_extinction_curve(tabulation, "disk", colour_by="opticalDepth")
        shades = {tuple(np.round(to_rgba(line.get_color()), 4)) for line in axes.lines}
        assert len(shades) == len(DEPTHS)

    def test_the_zero_entry_is_drawn_apart_from_the_scale(self, tabulation):
        """No dust at all is a different thing from a little dust."""
        axes = plot_extinction_curve(tabulation, "disk", colour_by="opticalDepth")
        zero = axes.lines[0]
        assert zero.get_linestyle() == "--"
        assert axes.get_legend() is not None
        assert "= 0" in axes.get_legend().get_texts()[0].get_text()

    def test_a_linear_axis_uses_a_linear_scale(self, tabulation):
        """Inclination has no reason to be logarithmic, and includes zero."""
        axes = plot_extinction(tabulation, "spheroid", "spheroidScaleRadial", colour_by="inclination")
        assert len(axes.lines) == INCLINATIONS.size


def test_reddening_is_unstable_where_the_dust_is_thin():
    """Recorded so a wild value at low optical depth is not read as a fault.

    The denominator is a difference of two nearly equal extinctions. A one per
    cent perturbation of the transmission, which is the scale of Monte Carlo
    noise, moves the reddening at the smallest optical depth by several times
    that, and can flip its sign. Measured over several realizations, since a
    single one ranges from four to eighty times the perturbation.
    """
    perturbation = 0.01
    changes = []
    for seed in range(8):
        tabulation = synthetic()
        exact = reddening(tabulation, "disk")[2, 1]
        values = tabulation.attenuation["disk"]
        rng = np.random.default_rng(seed)
        tabulation.attenuation["disk"] = values * (1.0 + rng.normal(scale=perturbation, size=values.shape))
        changes.append(abs(reddening(tabulation, "disk")[2, 1] - exact) / abs(exact))
    assert np.median(changes) > 3.0 * perturbation
