r"""Building the standard Hyperion dust files the compendium configurations use.

The Draine and Kim, Martin & Hendry grain models are not distributed as HDF5
files. They are shipped as tabulated optical properties in the ``hyperion-dust``
source package, with a ``setup.py build_dust`` command to turn them into the
files Hyperion reads. That command is written against ``distutils``, which
Python 3.12 removed, so it no longer runs; and the files it produces are a hard
prerequisite for every compendium configuration here.

This does the same work through the same Hyperion interfaces, so a tabulation
can be reproduced from the published source rather than from a file someone
happens to still have.

Get the source from https://pypi.org/project/hyperion-dust/ and unpack it; the
tabulations are under ``input/``.
"""

from pathlib import Path
from typing import Any

from .._hyperion import missing_hyperion

__all__ = ["EXTRAPOLATION_RANGE", "STANDARD_GRAINS", "build_library"]

#: The grain models the compendium configurations name, by the file they become,
#: with the reader each needs and where its tabulation sits in the source tree.
#: These are the five that ``hyperion-dust`` builds.
STANDARD_GRAINS: dict[str, tuple[str, str]] = {
    "d03_3.1_6.0_A": ("bhmie", "input/d03/d03_3.1_6.0_A/d03_3.1_6.0_A"),
    "d03_4.0_4.0_A": ("bhmie", "input/d03/d03_4.0_4.0_A/d03_4.0_4.0_A"),
    "d03_5.5_3.0_A": ("bhmie", "input/d03/d03_5.5_3.0_A/d03_5.5_3.0_A"),
    "kmh94_3.1_full": ("bhmie", "input/kmh94_full/kmh94_3.1_full/kmh94_3.1_full"),
    "kmh94_3.1_hg": ("whitney", "input/kmh94_hg/kmh94_3.1_hg"),
}

#: Wavelengths, in microns, the optical properties are extrapolated across, as
#: ``hyperion-dust`` does. Wide enough for any model here.
EXTRAPOLATION_RANGE = (1.0e-3, 1.0e7)


def build_library(
    source: str | Path,
    output: str | Path,
    names: list[str] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Build dust files from an unpacked ``hyperion-dust`` source tree.

    Parameters
    ----------
    source
        The unpacked source, the directory holding ``input/``.
    output
        Where to write the files. The compendium configurations expect
        ``hyperion-dust-0.1.0/dust_files`` relative to where they are run.
    names
        Which grain models to build; all of :data:`STANDARD_GRAINS` by default.
    overwrite
        Rebuild files which already exist.

    Returns
    -------
    The files written, in the order they were built.

    Raises
    ------
    FileNotFoundError
        If the source tree, or a tabulation within it, is not where it should
        be.
    KeyError
        If a name is not one of the standard grain models.
    ImportError
        If Hyperion is not installed.
    """
    try:
        from hyperion.dust import BHDust, TTsreDust
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise missing_hyperion("building the dust library") from error

    source, output = Path(source), Path(output)
    if not (source / "input").is_dir():
        raise FileNotFoundError(
            f"{source} does not look like an unpacked hyperion-dust source tree: "
            "it has no `input` directory. Download the source from "
            "https://pypi.org/project/hyperion-dust/ and unpack it."
        )
    for name in names or []:
        if name not in STANDARD_GRAINS:
            known = ", ".join(sorted(STANDARD_GRAINS))
            raise KeyError(f"unknown grain model {name!r}; known models are {known}")

    readers: dict[str, Any] = {"bhmie": BHDust, "whitney": TTsreDust}
    output.mkdir(parents=True, exist_ok=True)
    written = []
    for name in names or list(STANDARD_GRAINS):
        kind, location = STANDARD_GRAINS[name]
        destination = output / f"{name}.hdf5"
        if destination.exists() and not overwrite:
            continue
        tabulation = source / location
        # The tabulations are a prefix, with the properties in sibling files.
        if not any(tabulation.parent.glob(f"{tabulation.name}*")):
            raise FileNotFoundError(f"no tabulation for {name} at {tabulation}")
        dust = readers[kind](str(tabulation))
        dust.optical_properties.extrapolate_wav(*EXTRAPOLATION_RANGE)
        dust.write(str(destination))
        written.append(destination)
    return written
