r"""Galaxies as a set of named components, each optionally holding stars and dust.

The original code had one stellar component and one dust distribution wired in
by name, with a command line switch choosing whether the disk or the spheroid
was emitting. Here a galaxy is a sequence of :class:`Component`, each of which
may carry a stellar profile, a dust profile, both, or neither. Putting dust in
the spheroid as well as the disk is then a matter of giving the spheroid
component a dust profile, rather than a change to anything structural.

Optical depths are deliberately *not* held here. A galaxy is a geometry, and one
geometry is tabulated over a grid of optical depths; those belong to the model
being run, not to the shape of the thing being modelled.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass

from .geometry import Profile

__all__ = ["Component", "Galaxy"]


@dataclass(frozen=True)
class Component:
    """One named part of a galaxy, such as its disk or its spheroid.

    Parameters
    ----------
    name
        How the component is referred to in configurations, in optical depth
        mappings, and in the names of output datasets. Conventionally ``disk``
        or ``spheroid``.
    stellar
        The profile the component's stars follow, or ``None`` if it has none.
        Used as the probability of a photon being emitted from each cell.
    dust
        The profile the component's dust follows, or ``None`` if it has none.
        Scaled to a requested optical depth when a model is built.
    """

    name: str
    stellar: Profile | None = None
    dust: Profile | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a component needs a name")

    @property
    def emits(self) -> bool:
        """Whether this component has stars, and so can be a source of light."""
        return self.stellar is not None

    @property
    def attenuates(self) -> bool:
        """Whether this component has dust, and so contributes to the opacity."""
        return self.dust is not None


class Galaxy:
    """A collection of components with distinct names.

    Parameters
    ----------
    components
        The components, in the order their axes should appear in output. Names
        must be unique.

    Raises
    ------
    ValueError
        If no components are given, or if two share a name.
    """

    def __init__(self, components: Sequence[Component]) -> None:
        components = tuple(components)
        if not components:
            raise ValueError("a galaxy needs at least one component")
        names = [component.name for component in components]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"component names must be unique; repeated: {', '.join(duplicates)}")
        self.components = components

    def __iter__(self) -> Iterator[Component]:
        return iter(self.components)

    def __len__(self) -> int:
        return len(self.components)

    def __getitem__(self, name: str) -> Component:
        for component in self.components:
            if component.name == name:
                return component
        known = ", ".join(component.name for component in self.components)
        raise KeyError(f"no component named {name!r}; this galaxy has {known}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(component.name for component in self.components)

    @property
    def emitting(self) -> tuple[Component, ...]:
        """Components with stars, each of which gets its own tabulation."""
        return tuple(component for component in self.components if component.emits)

    @property
    def attenuating(self) -> tuple[Component, ...]:
        """Components with dust, each of which contributes an optical depth axis."""
        return tuple(component for component in self.components if component.attenuates)

    @property
    def extent_radial(self) -> float:
        """The largest radial scale in the galaxy, which sizes the grid."""
        return max(profile.extent_radial for profile in self._profiles())

    @property
    def extent_vertical(self) -> float:
        """The largest vertical scale in the galaxy, which sizes the grid.

        Note that a spheroid contributes its scale radius here as well as
        radially, so a spheroid larger than the disk's scale height sets the
        vertical extent -- as it did in the original.
        """
        return max(profile.extent_vertical for profile in self._profiles())

    def _profiles(self) -> Iterator[Profile]:
        for component in self.components:
            if component.stellar is not None:
                yield component.stellar
            if component.dust is not None:
                yield component.dust

    def __repr__(self) -> str:
        return f"Galaxy({list(self.components)!r})"
