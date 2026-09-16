r"""Dust matched to the models of Ferrara et al. (1999).

Ferrara et al. (1999; ApJS; 123; 437) used Henyey-Greenstein scattering with an
albedo, asymmetry and extinction curve from Gordon, Calzetti & Witt (1997; ApJ;
487; 625), for Milky Way and Small Magellanic Cloud grains. Those tabulations
live in ``data/gordon1997.csv``; this module turns them into a Hyperion dust
object.

The extinction curve is published relative to the V band, so it carries no
absolute scale. The original multiplied it by the V band opacity of a Draine
dust file, which is arbitrary -- densities are set by optical depth, so the
scale cancels -- but convenient, in that optical depths and dust masses then
mean the same thing across dust models. That reference is a parameter here
rather than a hard-coded file path, and defaults to leaving the curve as
published.
"""

import csv
from importlib import resources
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .properties import SPEED_OF_LIGHT_ANGSTROMS

__all__ = ["GRAIN_TYPES", "GordonTable", "build", "read_table", "write"]

#: Grain types, by their canonical name and the abbreviations the original used.
GRAIN_TYPES: dict[str, str] = {
    "milkyWay": "MilkyWay",
    "MW": "MilkyWay",
    "smallMagellanicCloud": "SmallMagellanicCloud",
    "SMC": "SmallMagellanicCloud",
}

#: Range over which the optical properties are extrapolated, in microns. From
#: ``hyperionDustFerrara.py``: wide enough to cover any wavelength the models
#: are tabulated at.
EXTRAPOLATION_RANGE = (0.005, 1000.0)


class GordonTable:
    """The Gordon et al. (1997) optical properties for one grain type.

    Attributes are ordered by *ascending frequency*, which is what Hyperion
    wants, and so by descending wavelength -- the reverse of how the table is
    published and stored.
    """

    def __init__(
        self,
        wavelength: NDArray[np.float64],
        albedo: NDArray[np.float64],
        asymmetry: NDArray[np.float64],
        extinction: NDArray[np.float64],
    ) -> None:
        order = np.argsort(wavelength)[::-1]
        self.wavelength = wavelength[order]
        self.albedo = albedo[order]
        self.asymmetry = asymmetry[order]
        self.extinction = extinction[order]

    @property
    def frequency(self) -> NDArray[np.float64]:
        """Frequencies in Hz, ascending."""
        return SPEED_OF_LIGHT_ANGSTROMS / self.wavelength

    def __len__(self) -> int:
        return self.wavelength.size


def read_table(grain_type: str) -> GordonTable:
    """Read the tabulated properties for a grain type.

    Parameters
    ----------
    grain_type
        ``milkyWay`` or ``smallMagellanicCloud``; the original abbreviations
        ``MW`` and ``SMC`` are accepted too.

    Raises
    ------
    KeyError
        If the grain type is not one of those tabulated.
    """
    try:
        suffix = GRAIN_TYPES[grain_type]
    except KeyError:
        known = ", ".join(sorted(set(GRAIN_TYPES)))
        raise KeyError(f"unknown grain type {grain_type!r}; known types are {known}") from None

    source = resources.files(__package__).joinpath("data/gordon1997.csv")
    with source.open("r", encoding="utf-8") as stream:
        rows = list(csv.DictReader(line for line in stream if not line.startswith("#")))

    def column(name: str) -> NDArray[np.float64]:
        return np.array([float(row[name]) for row in rows])

    return GordonTable(
        wavelength=column("wavelength"),
        albedo=column(f"albedo{suffix}"),
        asymmetry=column(f"asymmetry{suffix}"),
        extinction=column(f"extinction{suffix}"),
    )


def build(grain_type: str, reference_opacity: float = 1.0) -> Any:
    """Build a Hyperion dust object for a grain type.

    Parameters
    ----------
    grain_type
        ``milkyWay`` or ``smallMagellanicCloud``.
    reference_opacity
        The V band opacity to scale the published extinction curve onto. The
        default of one leaves it as published, relative to the V band. Pass the
        V band opacity of another dust model to put this one on the same
        absolute scale, which is what the original did with a Draine file.

    Raises
    ------
    ImportError
        If Hyperion is not installed.
    """
    try:
        from hyperion.dust import HenyeyGreensteinDust
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "building a dust model needs Hyperion, which is not installed; "
            "see the installation notes in the README"
        ) from error
    if reference_opacity <= 0.0:
        raise ValueError(f"reference opacity must be positive, got {reference_opacity}")

    table = read_table(grain_type)
    dust = HenyeyGreensteinDust(
        table.frequency,
        table.albedo,
        table.extinction * reference_opacity,
        table.asymmetry,
        # Allow full linear polarization, as the original did.
        np.ones(len(table)),
    )
    dust.optical_properties.extrapolate_wav(*EXTRAPOLATION_RANGE)
    return dust


def write(grain_type: str, file_name: str, reference_opacity: float = 1.0) -> None:
    """Build a dust model and write it to a Hyperion dust file."""
    build(grain_type, reference_opacity).write(file_name)
