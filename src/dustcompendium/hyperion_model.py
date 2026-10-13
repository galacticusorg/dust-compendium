r"""Turning a :class:`~dustcompendium.model.ModelSpec` into a Hyperion model.

This is the only module that needs Hyperion, and it is deliberately thin: the
grid, the dust density and the stellar emission map are all built in
:mod:`dustcompendium.model` with plain NumPy, and this hands them over along
with the settings the radiative transfer needs.

Those settings follow the original. The transfer is monochromatic, with no
initial iterations and photons killed on absorption, because the models are only
ever asked how much starlight escapes -- dust emission is not wanted and not
computed. Raytracing is on, and the output is a set of peel-off SEDs rather than
images.
"""

from typing import Any

import numpy as np

from ._hyperion import missing_hyperion
from .model import ModelSpec, dust_density, flat_spectrum, stellar_emission, viewing_angles

__all__ = ["build_model", "emitted_luminosity", "spectrum_for", "write_model"]

#: How far inside the dust's frequency range the source spectrum is kept. A
#: photon at a frequency the dust is not defined at aborts the solve.
SPECTRUM_MARGIN = 1.0e-6


def spectrum_for(dust: Any, points: int = 100) -> tuple[Any, Any]:
    """A flat source spectrum spanning the dust's own frequency range.

    Kept just inside that range. Sharing an endpoint with the dust is not safe:
    the two are computed separately and the last floating point digit need not
    agree, and a photon drawn a whisker outside aborts the solve. Hyperion exits
    zero when it aborts, so the failure shows up only as a solved model with no
    SEDs in it.
    """
    frequencies = dust.optical_properties.nu
    lowest = float(np.min(frequencies)) * (1.0 + SPECTRUM_MARGIN)
    highest = float(np.max(frequencies)) * (1.0 - SPECTRUM_MARGIN)
    return flat_spectrum((lowest, highest), points=points)


def build_model(
    spec: ModelSpec,
    dust: Any,
    opacity: float | None = None,
    sampling: str = "centre",
    dust_file: str | None = None,
    **grid_options: Any,
) -> Any:
    """Build a Hyperion model, ready to be written or run.

    Parameters
    ----------
    spec
        The geometry, optical depths, wavelengths and inclinations.
    dust
        A Hyperion dust object, from :func:`dustcompendium.dust.load_dust` or
        built by :mod:`dustcompendium.dust.ferrara`.
    opacity
        The V band opacity per unit dust mass used to normalize the dust
        densities. Taken from ``dust`` when not given, which is what you want
        unless deliberately mismatching them.
    sampling
        How to sample the dust density; see
        :func:`~dustcompendium.model.dust_density`.
    dust_file
        Write a reference to this path rather than embedding the grain
        properties in the model.

        Embedding makes each model self-contained, but the properties are the
        bulk of it: a Draine file is 12.8 MB, so a campaign of sixteen thousand
        models carries two hundred gigabytes of the same table copied over and
        over, against two gigabytes if it is referred to. The file then has to
        be present, at this path relative to wherever the model is solved, when
        it is solved. Give a path for a large campaign and leave it out for a
        self-contained one.
    **grid_options
        Passed through to :meth:`~dustcompendium.model.ModelSpec.grid`, for
        instance ``spacing="nested"`` or a different number of cells.

    Raises
    ------
    ImportError
        If Hyperion is not installed.
    """
    try:
        from hyperion.model import Model
        from hyperion.util.constants import lsun
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise missing_hyperion("building a radiative transfer model") from error

    from .dust import opacity_to_extinction

    if opacity is None:
        opacity = opacity_to_extinction(dust)

    grid = spec.grid(**grid_options)
    model = Model()
    model.set_seed(spec.seed)
    model.set_cylindrical_polar_grid(grid.radial_walls, grid.vertical_walls, grid.azimuthal_walls)

    model.add_density_grid(
        dust_density(spec.galaxy, grid, spec.optical_depths, opacity, sampling=sampling),
        dust_file if dust_file is not None else dust,
    )

    source = model.add_map_source()
    # An arbitrary normalization: only the ratio of emergent to emitted
    # luminosity is ever used, so it cancels.
    source.luminosity = lsun
    source.spectrum = spectrum_for(dust)
    source.map = stellar_emission(spec.galaxy, grid, spec.emitter)

    if spec.mrw:
        model.set_mrw(True, gamma=spec.mrw_gamma)

    model.set_monochromatic(True, wavelengths=spec.wavelengths)
    model.set_n_initial_iterations(0)
    model.set_kill_on_absorb(True)
    model.set_raytracing(True)

    image = model.add_peeled_images(image=False)
    image.set_wavelength_index_range(0, spec.wavelengths.size - 1)
    image.set_viewing_angles(*viewing_angles(spec.inclinations))
    image.set_uncertainties(True)

    model.set_n_photons(
        imaging_sources=spec.photons,
        imaging_dust=0,
        raytracing_sources=spec.photons,
        raytracing_dust=0,
    )
    return model


def write_model(
    spec: ModelSpec,
    dust: Any,
    file_name: str,
    opacity: float | None = None,
    sampling: str = "centre",
    overwrite: bool = True,
    dust_file: str | None = None,
    **grid_options: Any,
) -> None:
    """Build a model and write it to a Hyperion input file.

    The file is then solved by one of the Hyperion solver binaries, which are
    not installed by ``pip``; see the README.
    """
    model = build_model(spec, dust, opacity=opacity, sampling=sampling, dust_file=dust_file, **grid_options)
    # `copy=False` links to the grain properties instead of copying them in. The
    # link is relative to the model file, so the layout between the models and
    # the dust file has to be the same wherever they are solved.
    model.write(file_name, overwrite=overwrite, copy=dust_file is None)


def emitted_luminosity(spec: ModelSpec, **grid_options: Any) -> float:
    """Total stellar emission on the grid, in the units the map is built in.

    Useful as a check that the grid is large enough to hold the component: it
    should be close to the profile's total mass.
    """
    grid = spec.grid(**grid_options)
    return float(np.sum(stellar_emission(spec.galaxy, grid, spec.emitter)))
