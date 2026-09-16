r"""Reading grain optical properties from a Hyperion dust file.

Only one number is needed from a dust file to build a model: the opacity to
extinction in the V band. Densities are set by requiring a given optical depth,
and the dust files give opacities per unit *dust* mass, so the normalization
those densities carry is a dust density.

Importing this module does not require Hyperion; calling into it does.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - import only for type checking
    pass

__all__ = [
    "SPEED_OF_LIGHT_ANGSTROMS",
    "V_BAND_WAVELENGTH",
    "load_dust",
    "opacity_to_extinction",
]

#: The V band, in microns. Hyperion's dust files are tabulated in microns.
V_BAND_WAVELENGTH = 0.55

#: Speed of light in Angstroms per second, for converting the tabulations that
#: are published against wavelength into the frequencies Hyperion wants.
SPEED_OF_LIGHT_ANGSTROMS = 2.998e18


def load_dust(file_name: str) -> Any:
    """Read a Hyperion spherical dust file.

    Parameters
    ----------
    file_name
        Path to a dust file, such as one of those built by the ``hyperion-dust``
        package.

    Raises
    ------
    ImportError
        If Hyperion is not installed. See the installation notes in the README:
        no released version can build models on NumPy 2, so it has to come from
        source for now.
    """
    try:
        from hyperion.dust import SphericalDust
    except ImportError as error:  # pragma: no cover - depends on the environment
        raise ImportError(
            "reading a dust file needs Hyperion, which is not installed; "
            "see the installation notes in the README"
        ) from error
    dust = SphericalDust()
    dust.read(file_name)
    return dust


def opacity_to_extinction(dust: Any, wavelength: float = V_BAND_WAVELENGTH) -> float:
    r"""The opacity to extinction per unit dust mass at a wavelength.

    Parameters
    ----------
    dust
        A Hyperion dust object, from :func:`load_dust` or built in this package.
    wavelength
        Wavelength in microns, the V band by default.

    Returns
    -------
    :math:`\kappa`, in units inverse to a column density, so that
    :math:`\tau = \kappa \Sigma`.
    """
    return float(dust.optical_properties.interp_chi_wav(wavelength))
