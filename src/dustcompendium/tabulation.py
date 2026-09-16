r"""The tabulation itself, and the file it is written to.

The file format is a contract. Galacticus reads these files in
``source/dust/attenuation/atlas_compendium.F90``, which checks the shape of
every array it reads against the axes, on the grounds that "a transposed read
would otherwise show up much later as quietly wrong attenuation". The layout
below is what it expects, and the tests assert it.

Two versions exist. Version 1 is the published layout: a single optical depth,
belonging to the disk, and a spheroid tabulated against its size. Version 2 is a
strict superset, written when the campaign has more axes than that -- notably
dust in the spheroid as well as the disk -- and keeps every version 1 name
meaning what it meant. A version 2 file is correctly *rejected* by the current
Fortran class, through those same shape checks, rather than quietly misread.

Arrays are stored so that the axes read, in C order, as wavelength, inclination,
then the campaign's own axes in order. That is the published layout, and it
falls out of the order the axes are expanded in rather than being imposed here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray

__all__ = ["Tabulation", "axis_dataset_name", "read_tabulation"]

#: The axis names version 1 uses, which are not derivable from the axis itself.
#: Anything outside this table is named from its kind and component.
LEGACY_AXIS_NAMES = {
    "opticalDepth:disk": "opticalDepth",
    "opticalDepth:spheroid": "spheroidOpticalDepth",
    "scaleRadial:spheroid": "spheroidScaleRadial",
}

#: The axes a version 1 file has, in order. Anything else makes it version 2.
VERSION_ONE_AXES = ("opticalDepth:disk", "scaleRadial:spheroid")

#: The emitters a version 1 file has.
VERSION_ONE_EMITTERS = ("disk", "spheroid")


def axis_dataset_name(name: str) -> str:
    """The dataset an axis is stored as.

    Keeps the published names, which are not a pattern -- the disk's optical
    depth is plain ``opticalDepth``, and the spheroid's scale is
    ``spheroidScaleRadial`` -- and falls back to a derived name otherwise.
    """
    if name in LEGACY_AXIS_NAMES:
        return LEGACY_AXIS_NAMES[name]
    kind, _, component = name.partition(":")
    if not component:
        return kind
    return f"{component}{kind[0].upper()}{kind[1:]}"


@dataclass
class Tabulation:
    """Attenuation as a function of wavelength, inclination and the campaign's axes.

    Parameters
    ----------
    label
        The campaign's label, recorded in the file.
    wavelengths
        In microns, as tabulated.
    inclinations
        In degrees, from face-on to edge-on.
    axis_names
        The campaign's axes, in the order they index the arrays.
    axis_values
        The values of each axis, by name.
    emitter_axes
        Which axes each emitter is tabulated against, by emitter. An emitter's
        arrays are indexed by wavelength, inclination, then these.
    attenuation
        The fraction of light escaping, by emitter.
    uncertainty
        Monte Carlo uncertainty on it, by emitter. Galacticus does not read
        these -- they are half of every published file -- but they are what says
        whether a tabulation is converged.
    extrapolation
        Coefficients of ``exp(c0 + c1 ln tau)`` beyond the largest optical
        depth, by emitter, indexed by coefficient, wavelength, inclination and
        then every axis except the one extrapolated along.
    residual
        Root mean square residual of that fit, in the log, by emitter. Not part
        of the published format: it is what lets the power-law assumption be
        checked rather than trusted.
    opacity
        Opacity to extinction per unit dust mass in the V band. Galacticus uses
        its presence to recognise the file.
    metadata
        Extra attributes, recorded as given.
    """

    label: str
    wavelengths: NDArray[np.float64]
    inclinations: NDArray[np.float64]
    axis_names: tuple[str, ...]
    axis_values: Mapping[str, NDArray[np.float64]]
    emitter_axes: Mapping[str, tuple[str, ...]]
    attenuation: Mapping[str, NDArray[np.float64]]
    uncertainty: Mapping[str, NDArray[np.float64]]
    extrapolation: Mapping[str, NDArray[np.float64]]
    residual: Mapping[str, NDArray[np.float64]]
    opacity: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def emitters(self) -> tuple[str, ...]:
        return tuple(self.attenuation)

    @property
    def format_version(self) -> int:
        """1 for the published layout, 2 for anything beyond it."""
        if self.axis_names == VERSION_ONE_AXES and tuple(self.emitters) == VERSION_ONE_EMITTERS:
            return 1
        return 2

    def expected_shape(self, emitter: str) -> tuple[int, ...]:
        """The shape an emitter's attenuation array must have."""
        return (
            self.wavelengths.size,
            self.inclinations.size,
            *(self.axis_values[name].size for name in self.emitter_axes[emitter]),
        )

    def check(self) -> None:
        """Verify every array has the shape its axes imply.

        Raises
        ------
        ValueError
            Naming the array and both shapes. This is the same check Galacticus
            makes when reading, made here so a bad file is never written.
        """
        for emitter in self.emitters:
            expected = self.expected_shape(emitter)
            for name, arrays in (
                ("attenuation", self.attenuation),
                ("uncertainty", self.uncertainty),
            ):
                shape = tuple(np.shape(arrays[emitter]))
                if shape != expected:
                    raise ValueError(
                        f"{name} for {emitter!r} has shape {shape}, but its axes imply {expected}"
                    )
            extrapolated = tuple(
                self.axis_values[name].size
                for name in self.emitter_axes[emitter]
                if name != self.extrapolation_axis
            )
            expected_extrapolation = (
                2,
                self.wavelengths.size,
                self.inclinations.size,
                *extrapolated,
            )
            shape = tuple(np.shape(self.extrapolation[emitter]))
            if shape != expected_extrapolation:
                raise ValueError(
                    f"extrapolation for {emitter!r} has shape {shape}, "
                    f"but its axes imply {expected_extrapolation}"
                )

    @property
    def extrapolation_axis(self) -> str:
        """Which optical depth the high-depth extrapolation runs along.

        The disk's, following the published files. With dust in more than one
        component the choice is not forced, and this one keeps the coefficients
        tabulated against every other axis, so a spheroid's own optical depth
        stays bounded by the table rather than being extrapolated through.
        """
        return VERSION_ONE_AXES[0]

    def write(self, path: str) -> None:
        """Write the tabulation to an HDF5 file."""
        self.check()
        with h5py.File(path, "w") as handle:
            handle.attrs["formatVersion"] = self.format_version
            handle.attrs["label"] = self.label
            handle.attrs["opacity"] = float(self.opacity)
            handle.attrs["timeStamp"] = datetime.now(UTC).isoformat()
            handle.attrs["extrapolationAxis"] = axis_dataset_name(self.extrapolation_axis)
            for name, value in self.metadata.items():
                handle.attrs[name] = value

            handle.create_dataset("wavelength", data=self.wavelengths)
            handle.create_dataset("inclination", data=self.inclinations)
            for name in self.axis_names:
                dataset = handle.create_dataset(axis_dataset_name(name), data=self.axis_values[name])
                # The dataset names are the published ones, which are not a
                # pattern and cannot always be inverted; record the axis this
                # came from so that reading a file back is unambiguous.
                dataset.attrs["axis"] = name

            for emitter in self.emitters:
                suffix = emitter[0].upper() + emitter[1:]
                axes = ["wavelength", "inclination"] + [
                    axis_dataset_name(name) for name in self.emitter_axes[emitter]
                ]
                for prefix, arrays in (
                    ("attenuation", self.attenuation),
                    ("attenuationUncertainty", self.uncertainty),
                ):
                    dataset = handle.create_dataset(f"{prefix}{suffix}", data=arrays[emitter])
                    dataset.attrs["axes"] = axes
                coefficients = handle.create_dataset(
                    f"extrapolationCoefficients{suffix}", data=self.extrapolation[emitter]
                )
                coefficients.attrs["axes"] = ["coefficient"] + [
                    axis for axis in axes if axis != axis_dataset_name(self.extrapolation_axis)
                ]
                residuals = handle.create_dataset(
                    f"extrapolationResidual{suffix}", data=self.residual[emitter]
                )
                residuals.attrs["axes"] = coefficients.attrs["axes"][1:]


def read_tabulation(path: str) -> Tabulation:
    """Read a tabulation written by :meth:`Tabulation.write`.

    Files predating the ``formatVersion`` attribute are version 1, which is what
    every published tabulation is.
    """
    with h5py.File(path, "r") as handle:
        if "opacity" not in handle.attrs:
            raise ValueError(f"{path} has no opacity attribute, so is not a dust compendium tabulation")
        metadata = {
            name: value
            for name, value in handle.attrs.items()
            if name not in ("formatVersion", "label", "opacity", "timeStamp", "extrapolationAxis")
        }
        emitters = [
            name[len("attenuation") :]
            for name in handle
            if name.startswith("attenuation") and not name.startswith("attenuationUncertainty")
        ]
        # Map the stored dataset names back to the axes they came from.
        campaign_names = {
            name: str(handle[name].attrs["axis"]) for name in handle if "axis" in handle[name].attrs
        }
        axis_names: list[str] = []
        emitter_axes: dict[str, tuple[str, ...]] = {}
        stored: dict[str, NDArray[np.float64]] = {}
        attenuation, uncertainty, extrapolation, residual = {}, {}, {}, {}
        for suffix in emitters:
            emitter = suffix[0].lower() + suffix[1:]
            dataset = handle[f"attenuation{suffix}"]
            stored_axes = [str(axis) for axis in dataset.attrs["axes"]][2:]
            axes = [campaign_names.get(axis, axis) for axis in stored_axes]
            emitter_axes[emitter] = tuple(axes)
            for axis, source in zip(axes, stored_axes, strict=True):
                if axis not in axis_names:
                    axis_names.append(axis)
                    stored[axis] = np.array(handle[source])
            attenuation[emitter] = np.array(dataset)
            uncertainty[emitter] = np.array(handle[f"attenuationUncertainty{suffix}"])
            extrapolation[emitter] = np.array(handle[f"extrapolationCoefficients{suffix}"])
            name = f"extrapolationResidual{suffix}"
            residual[emitter] = np.array(handle[name]) if name in handle else np.array([])
        return Tabulation(
            label=str(handle.attrs.get("label", "")),
            wavelengths=np.array(handle["wavelength"]),
            inclinations=np.array(handle["inclination"]),
            axis_names=tuple(axis_names),
            axis_values=stored,
            emitter_axes=emitter_axes,
            attenuation=attenuation,
            uncertainty=uncertainty,
            extrapolation=extrapolation,
            residual=residual,
            opacity=float(handle.attrs["opacity"]),
            metadata=metadata,
        )
