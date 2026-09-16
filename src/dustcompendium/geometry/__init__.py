"""Density profiles for the stellar and dust components of a galaxy."""

from .base import Profile
from .disk import (
    ExponentialDisk,
    ExponentialVertical,
    SechSquaredVertical,
    VerticalStructure,
)
from .spheroid import HernquistSpheroid, JaffeSpheroid, Spheroid

#: Vertical structures, by the names the original configurations used.
VERTICAL_STRUCTURES: dict[str, type[VerticalStructure]] = {
    "exponential": ExponentialVertical,
    "sechSquared": SechSquaredVertical,
}

#: Spheroid profiles, by the names the original configurations used.
SPHEROID_PROFILES: dict[str, type[Spheroid]] = {
    "hernquist": HernquistSpheroid,
    "jaffe": JaffeSpheroid,
}


def vertical_structure(name: str, scale_height: float) -> VerticalStructure:
    """Build a vertical structure by name.

    Raises
    ------
    KeyError
        If the name is not a known vertical structure. The message lists those
        that are.
    """
    try:
        factory = VERTICAL_STRUCTURES[name]
    except KeyError:
        known = ", ".join(sorted(VERTICAL_STRUCTURES))
        raise KeyError(f"unknown vertical structure {name!r}; known structures are {known}") from None
    return factory(scale_height)


def spheroid_profile(name: str, scale_radial: float, truncation: float | None = None) -> Spheroid:
    """Build a spheroid profile by name.

    Parameters
    ----------
    name
        ``hernquist`` or ``jaffe``.
    scale_radial
        The scale radius.
    truncation
        Radius, in scale radii, beyond which the density is zero. This is the
        original's ``spheroidCutOff``, which it parsed and then never used.

    Raises
    ------
    KeyError
        If the name is not a known spheroid profile. The message lists those
        that are.
    """
    try:
        factory = SPHEROID_PROFILES[name]
    except KeyError:
        known = ", ".join(sorted(SPHEROID_PROFILES))
        raise KeyError(f"unknown spheroid profile {name!r}; known profiles are {known}") from None
    return factory(scale_radial, truncation)


__all__ = [
    "SPHEROID_PROFILES",
    "VERTICAL_STRUCTURES",
    "ExponentialDisk",
    "ExponentialVertical",
    "HernquistSpheroid",
    "JaffeSpheroid",
    "Profile",
    "SechSquaredVertical",
    "Spheroid",
    "VerticalStructure",
    "spheroid_profile",
    "vertical_structure",
]
