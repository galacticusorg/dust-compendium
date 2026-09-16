r"""Comparing a tabulation against a published one.

The point of reference is the atlas of Ferrara et al. (1999; ApJS; 123; 437),
which reproducing was how the method was checked in the first place. Galacticus
ships it converted to the same layout this package writes, as
``datasets/static/dust/atlasFerrara2000/``, so the two can be compared directly.

Two things have to be reconciled before they can be. Ferrara et al. tabulate
against the spheroid's *effective* radius while this package uses the scale
radius, which is 1.16 times larger, and they publish wavelengths in Angstroms
rather than microns. Both are handled here rather than being left to whoever is
doing the comparing.

The atlas is quoted to two decimal places -- some of its entries are 1.01, which
is a transmission above one -- so agreement closer than about 0.01 in
transmission cannot be demonstrated against it, however good the models are.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from numpy.typing import NDArray

from .tabulation import Tabulation

__all__ = [
    "ANGSTROMS_PER_MICRON",
    "ATLAS_QUANTIZATION",
    "SCALE_RADIUS_PER_EFFECTIVE_RADIUS",
    "Comparison",
    "compare_with_atlas",
    "read_atlas",
]

#: Ferrara et al. use a Jaffe spheroid, whose scale radius is this multiple of
#: the effective radius they tabulate against.
SCALE_RADIUS_PER_EFFECTIVE_RADIUS = 1.16

ANGSTROMS_PER_MICRON = 1.0e4

#: The atlas is published to two decimal places, so this is the floor on any
#: disagreement with it.
ATLAS_QUANTIZATION = 0.005


@dataclass(frozen=True)
class Atlas:
    """A published attenuation atlas, on its own axes.

    Wavelengths are converted to microns and spheroid sizes to scale radii, so
    that the axes mean the same thing as a tabulation's.
    """

    wavelengths: NDArray[np.float64]
    inclinations: NDArray[np.float64]
    optical_depths: NDArray[np.float64]
    scale_radii: NDArray[np.float64]
    attenuation: Mapping[str, NDArray[np.float64]]
    metadata: dict


def read_atlas(path: str | Path) -> Atlas:
    """Read a published atlas, putting its axes into this package's units."""
    with h5py.File(str(path), "r") as handle:
        return Atlas(
            wavelengths=np.array(handle["wavelength"]) / ANGSTROMS_PER_MICRON,
            inclinations=np.array(handle["inclination"]),
            optical_depths=np.array(handle["opticalDepth"]),
            scale_radii=np.array(handle["spheroidScaleRadial"]) * SCALE_RADIUS_PER_EFFECTIVE_RADIUS,
            attenuation={
                "disk": np.array(handle["attenuationDisk"]),
                "spheroid": np.array(handle["attenuationSpheroid"]),
            },
            metadata=dict(handle.attrs),
        )


@dataclass(frozen=True)
class Comparison:
    """How closely a tabulation reproduces an atlas, for one emitter."""

    emitter: str
    computed: NDArray[np.float64]
    published: NDArray[np.float64]

    @property
    def difference(self) -> NDArray[np.float64]:
        """Computed minus published, in transmission."""
        return self.computed - self.published

    @property
    def worst(self) -> float:
        return float(np.nanmax(np.abs(self.difference)))

    @property
    def median(self) -> float:
        return float(np.nanmedian(np.abs(self.difference)))

    def within(self, tolerance: float) -> NDArray[np.bool_]:
        """Which entries agree to a given absolute tolerance."""
        return np.abs(self.difference) <= tolerance

    def fraction_within(self, tolerance: float) -> float:
        return float(np.mean(self.within(tolerance)))

    def summary(self) -> str:
        return (
            f"{self.emitter}: median |difference| {self.median:.4f}, "
            f"worst {self.worst:.4f}, "
            f"{100.0 * self.fraction_within(0.02):.1f}% within 0.02"
        )


def compare_with_atlas(tabulation: Tabulation, atlas: Atlas, emitter: str) -> Comparison:
    """Line a tabulation up with an atlas and take the difference.

    The tabulation's axes must contain the atlas's, which they do when it was
    computed from a matched configuration. The optical depth of zero a
    tabulation carries, for the model with no dust, has no counterpart in the
    atlas and is dropped.

    Raises
    ------
    ValueError
        If an axis of the atlas is not present in the tabulation, saying which.
    """
    axes = tabulation.emitter_axes[emitter]
    wanted: dict[str, NDArray[np.float64]] = {
        "wavelength": atlas.wavelengths,
        "inclination": atlas.inclinations,
        "opticalDepth:disk": atlas.optical_depths,
    }
    if "scaleRadial:spheroid" in axes:
        wanted["scaleRadial:spheroid"] = atlas.scale_radii

    available = {
        "wavelength": tabulation.wavelengths,
        "inclination": tabulation.inclinations,
        **{name: tabulation.axis_values[name] for name in axes},
    }
    selection = []
    for name, values in wanted.items():
        if name not in available:
            raise ValueError(f"the tabulation has no {name!r} axis to compare against")
        indices = []
        for value in values:
            match = np.flatnonzero(np.isclose(available[name], value, rtol=1.0e-4))
            if match.size == 0:
                raise ValueError(
                    f"the tabulation's {name!r} axis has no entry at {value:g}; "
                    "it was not computed from a matched configuration"
                )
            indices.append(int(match[0]))
        selection.append(np.asarray(indices))

    computed = tabulation.attenuation[emitter]
    for position, indices in enumerate(selection):
        computed = np.take(computed, indices, axis=position)
    return Comparison(emitter, computed, atlas.attenuation[emitter])
