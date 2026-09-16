r"""Figures made from a tabulation.

The original had nine plotting scripts, which were three quantities --
extinction in the V band, the reddening :math:`R_\mathrm{V}`, and the extinction
curve itself -- each plotted against three things: inclination, optical depth
and spheroid size. They were near-copies of one another, and had drifted: three
of them read datasets under names the driver had stopped writing, so they would
not have run against anything it produced.

Here there is one implementation, parameterized by what to plot and what to plot
it against, which also covers combinations the nine did not: any quantity
against any axis of the tabulation, at any wavelength.

Needs matplotlib, which is an optional dependency -- install the ``plots`` extra.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .tabulation import Tabulation, axis_dataset_name

__all__ = [
    "B_BAND",
    "V_BAND",
    "coordinates",
    "extinction",
    "plot_extinction",
    "plot_extinction_curve",
    "reddening",
]

#: The B and V bands, in microns, as the original used them.
B_BAND = 0.44
V_BAND = 0.55

#: Axis labels, in the LaTeX the original's titles used.
LABELS = {
    "inclination": r"Inclination; $i\,[^\circ]$",
    "opticalDepth": r"Optical depth; $\tau_\mathrm{V,0}$",
    "spheroidOpticalDepth": r"Spheroid optical depth; $\tau_\mathrm{V,s}$",
    "spheroidScaleRadial": r"Spheroid size; $r_\mathrm{s}/r_\mathrm{d}$",
    "wavelength": r"Wavelength; $\lambda\,[\mu\mathrm{m}]$",
    "extinction": r"Extinction; $A_\lambda$",
    "reddening": r"Reddening; $R_\mathrm{V}$",
}

#: Axes which are naturally read logarithmically.
LOGARITHMIC = {"opticalDepth", "spheroidOpticalDepth", "spheroidScaleRadial", "wavelength"}


def label_for(name: str) -> str:
    """A readable axis label, falling back to the name itself."""
    return LABELS.get(axis_dataset_name(name), name)


def coordinates(tabulation: Tabulation, emitter: str) -> dict[str, NDArray[np.float64]]:
    """What an emitter's attenuation is indexed by, after wavelength.

    In array order, so the result can be zipped against the trailing axes.
    """
    return {
        "inclination": tabulation.inclinations,
        **{name: tabulation.axis_values[name] for name in tabulation.emitter_axes[emitter]},
    }


def _resolve(tabulation: Tabulation, emitter: str, name: str) -> str:
    """Accept either an axis's own name or the name it is stored under."""
    available = coordinates(tabulation, emitter)
    if name in available:
        return name
    for candidate in available:
        if axis_dataset_name(candidate) == name:
            return candidate
    known = ", ".join(available)
    raise KeyError(f"{emitter!r} is not tabulated against {name!r}; it has {known}")


def extinction(tabulation: Tabulation, emitter: str, wavelength: float = V_BAND) -> NDArray[np.float64]:
    r"""Extinction in magnitudes at a wavelength, :math:`A_\lambda = -2.5\log_{10} T`.

    The transmission is converted to magnitudes *before* interpolating, as the
    original did, and interpolated in the logarithm of wavelength since the grid
    is logarithmic. Interpolating the transmission instead would be noticeably
    worse where it is small: it falls exponentially with optical depth, while
    the extinction is close to a power law in wavelength.

    Raises
    ------
    ValueError
        If the wavelength is outside the tabulated range.
    """
    wavelengths = tabulation.wavelengths
    if not wavelengths.min() <= wavelength <= wavelengths.max():
        raise ValueError(
            f"{wavelength} micron is outside the tabulated range "
            f"{wavelengths.min():g} to {wavelengths.max():g}"
        )
    attenuation = tabulation.attenuation[emitter]
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitudes = -2.5 * np.log10(attenuation)
    order = np.argsort(wavelengths)
    abscissa = np.log(wavelengths[order])
    flattened = magnitudes[order].reshape(wavelengths.size, -1)
    interpolated = np.array([np.interp(np.log(wavelength), abscissa, column) for column in flattened.T])
    return interpolated.reshape(magnitudes.shape[1:])


def reddening(
    tabulation: Tabulation,
    emitter: str,
    blue: float = B_BAND,
    visual: float = V_BAND,
) -> NDArray[np.float64]:
    r"""The ratio of total to selective extinction, :math:`A_V / (A_B - A_V)`.

    Undefined where the two extinctions are equal, which is where there is no
    dust; those entries come back as not-a-number rather than an error.

    It is also worth little at *small* optical depth. The denominator is a
    difference of two nearly equal extinctions, so where the dust is thin it is
    comparable to the Monte Carlo noise on either, and the ratio scatters wildly
    -- it can come out negative. That is the quantity behaving as it must with
    finite photons, not a fault in the tabulation, but it means a reddening
    plotted from the low optical depth end says more about the photon count than
    about the dust.
    """
    visual_extinction = extinction(tabulation, emitter, visual)
    blue_extinction = extinction(tabulation, emitter, blue)
    selective = blue_extinction - visual_extinction
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(selective != 0.0, visual_extinction / selective, np.nan)


def _select(
    tabulation: Tabulation,
    emitter: str,
    values: NDArray[np.float64],
    against: str,
    colour_by: str | None,
    fixed: Mapping[str, int] | None,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64] | None]:
    """Reduce an array over the coordinates not being plotted."""
    available = list(coordinates(tabulation, emitter))
    fixed = {_resolve(tabulation, emitter, name): index for name, index in (fixed or {}).items()}
    chosen = [against] + ([colour_by] if colour_by else [])
    index: list[Any] = []
    for position, name in enumerate(available):
        if name in chosen:
            index.append(slice(None))
        else:
            index.append(fixed.get(name, 0))
        del position
    reduced = values[tuple(index)]
    # The remaining axes are in the order they appear, which need not be the
    # order they were asked for.
    remaining = [name for name in available if name in chosen]
    if colour_by and remaining.index(against) > remaining.index(colour_by):
        reduced = reduced.T
    abscissa = coordinates(tabulation, emitter)[against]
    family = coordinates(tabulation, emitter)[colour_by] if colour_by else None
    return reduced, abscissa, family


def plot_extinction(
    tabulation: Tabulation,
    emitter: str,
    against: str,
    quantity: str = "extinction",
    wavelength: float = V_BAND,
    colour_by: str | None = None,
    fixed: Mapping[str, int] | None = None,
    axes: Any = None,
) -> Any:
    r"""Plot extinction or reddening against one of the tabulated coordinates.

    Parameters
    ----------
    tabulation
        What to plot.
    emitter
        Which component's light.
    against
        The coordinate along the horizontal axis: ``inclination`` or any axis
        the emitter is tabulated against, by either name.
    quantity
        ``extinction`` for :math:`A_\lambda`, or ``reddening`` for
        :math:`R_\mathrm{V}`.
    wavelength
        Which wavelength to take the extinction at, in microns. Ignored for the
        reddening, which is defined by the B and V bands.
    colour_by
        Draw a family of curves over this coordinate instead of a single one.
    fixed
        Index to hold each remaining coordinate at. Anything unnamed is held at
        its first value, which for an optical depth is zero -- no dust.
    axes
        Draw onto these axes rather than making a figure.

    Returns
    -------
    The axes drawn on.
    """
    import matplotlib.pyplot as plt

    against = _resolve(tabulation, emitter, against)
    colour_by = _resolve(tabulation, emitter, colour_by) if colour_by else None
    if quantity == "extinction":
        values = extinction(tabulation, emitter, wavelength)
        ylabel = rf"Extinction; $A_{{{wavelength:g}\,\mu\mathrm{{m}}}}$"
    elif quantity == "reddening":
        values = reddening(tabulation, emitter)
        ylabel = LABELS["reddening"]
    else:
        raise ValueError(f"quantity must be 'extinction' or 'reddening', got {quantity!r}")

    reduced, abscissa, family = _select(tabulation, emitter, values, against, colour_by, fixed)
    axes = axes or plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)[1]
    if family is None:
        axes.plot(abscissa, reduced, linewidth=2.0)
    else:
        _draw_family(axes, abscissa, reduced, family, colour_by)
    axes.set_xlabel(label_for(against))
    axes.set_ylabel(ylabel)
    if axis_dataset_name(against) in LOGARITHMIC and np.all(abscissa > 0.0):
        axes.set_xscale("log")
    axes.set_title(f"{tabulation.label} ({emitter})", fontsize="small")
    return axes


def plot_extinction_curve(
    tabulation: Tabulation,
    emitter: str,
    colour_by: str | None = None,
    fixed: Mapping[str, int] | None = None,
    axes: Any = None,
) -> Any:
    r"""Plot the extinction curve, :math:`A_\lambda` against wavelength.

    With ``colour_by`` this is the original's three ``extinctionCurveVs``
    scripts: a family of curves over inclination, optical depth or spheroid size.
    """
    import matplotlib.pyplot as plt

    colour_by = _resolve(tabulation, emitter, colour_by) if colour_by else None
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitudes = -2.5 * np.log10(tabulation.attenuation[emitter])

    available = list(coordinates(tabulation, emitter))
    resolved = {_resolve(tabulation, emitter, name): index for name, index in (fixed or {}).items()}
    index: list[Any] = [slice(None)]
    for name in available:
        index.append(slice(None) if name == colour_by else resolved.get(name, 0))
    reduced = magnitudes[tuple(index)]

    axes = axes or plt.subplots(figsize=(6.0, 4.5), constrained_layout=True)[1]
    if colour_by is None:
        axes.plot(tabulation.wavelengths, reduced, linewidth=2.0)
    else:
        _draw_family(
            axes,
            tabulation.wavelengths,
            reduced,
            coordinates(tabulation, emitter)[colour_by],
            colour_by,
        )
    axes.set_xlabel(LABELS["wavelength"])
    axes.set_ylabel(LABELS["extinction"])
    axes.set_xscale("log")
    axes.set_title(f"{tabulation.label} ({emitter})", fontsize="small")
    return axes


def _draw_family(axes: Any, abscissa: NDArray, values: NDArray, family: Sequence[float], name: str) -> None:
    """Draw one curve per value of a coordinate, coloured along it.

    An optical depth axis carries an exact zero, for the model with no dust.
    That cannot go on a logarithmic colour scale, and putting the whole axis on
    a linear one instead crowds every curve below the largest depth into one
    colour -- with depths of 0, 0.1, 1 and 10, three of the four come out
    indistinguishable. So the positive values are scaled logarithmically, and
    the zero is drawn dashed and in grey rather than given a colour. It would
    otherwise land on the bottom of the scale alongside the smallest depth,
    which is a different thing entirely: no dust at all, rather than a little.
    """
    import matplotlib.pyplot as plt
    from matplotlib import colors

    family = np.asarray(family, dtype=float)
    positive = family[family > 0.0]
    logarithmic = axis_dataset_name(name) in LOGARITHMIC and positive.size > 1
    if logarithmic:
        normalize = colors.LogNorm(vmin=positive.min(), vmax=positive.max())
    else:
        normalize = colors.Normalize(vmin=family.min(), vmax=family.max())
    colourmap = plt.get_cmap("viridis")

    labelled = False
    for position, value in enumerate(family):
        if logarithmic and value <= 0.0:
            axes.plot(
                abscissa,
                values[:, position],
                color="0.5",
                linestyle="--",
                linewidth=1.5,
                label=None if labelled else f"{label_for(name).split(';')[0]} = 0",
            )
            labelled = True
        else:
            axes.plot(
                abscissa,
                values[:, position],
                color=colourmap(normalize(value)),
                linewidth=1.5,
            )

    axes.figure.colorbar(
        plt.cm.ScalarMappable(norm=normalize, cmap=colourmap), ax=axes, label=label_for(name)
    )
    if labelled:
        axes.legend(fontsize="small", frameon=False)
