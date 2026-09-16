"""Shared fixtures, and skipping for optional dependencies and reference data."""

import os
import shutil
from pathlib import Path

import pytest

#: Where the Ferrara et al. (1999) atlas lives when Galacticus is installed.
#: Set ``DUST_COMPENDIUM_REFERENCE`` to point somewhere else.
ATLAS_LOCATIONS = (
    "$DUST_COMPENDIUM_REFERENCE",
    "$GALACTICUS_DATA_PATH/static/dust/atlasFerrara2000",
    "~/Galacticus/datasets/static/dust/atlasFerrara2000",
)


def reference_directory() -> Path | None:
    """The first directory of reference tabulations which exists, if any.

    These are published data, far too large to vendor and not reproducible in a
    test run, so anything needing them is skipped when they are absent. CI has
    none of them.
    """
    for location in ATLAS_LOCATIONS:
        expanded = Path(os.path.expandvars(location)).expanduser()
        if "$" not in str(expanded) and expanded.is_dir():
            return expanded
    return None


def pytest_collection_modifyitems(config, items):
    """Skip tests needing Hyperion, or a solver binary, when they are absent.

    Hyperion is not a dependency of this package -- no released version can
    build models on NumPy 2 -- and the Fortran solvers are not installable with
    pip at all. Both are therefore optional, and CI runs without either.
    """
    try:
        import hyperion  # noqa: F401

        has_hyperion = True
    except ImportError:
        has_hyperion = False
    has_solver = shutil.which("hyperion_cyl") is not None
    has_reference = reference_directory() is not None

    skip_hyperion = pytest.mark.skip(reason="the Hyperion Python package is not installed")
    skip_solver = pytest.mark.skip(reason="no hyperion_cyl solver binary on PATH")
    skip_reference = pytest.mark.skip(
        reason="no published reference tabulations available; see tests/conftest.py"
    )
    for item in items:
        if "hyperion" in item.keywords and not has_hyperion:
            item.add_marker(skip_hyperion)
        if "solver" in item.keywords and not (has_solver and has_hyperion):
            item.add_marker(skip_solver)
        if "reference" in item.keywords and not has_reference:
            item.add_marker(skip_reference)


@pytest.fixture
def solver_path():
    """Path to the serial cylindrical solver."""
    path = shutil.which("hyperion_cyl")
    if path is None:  # pragma: no cover - guarded by the marker
        pytest.skip("no hyperion_cyl solver binary on PATH")
    return path


@pytest.fixture
def reference_atlas():
    """The Ferrara et al. (1999) atlas for Milky Way grains, if it is available."""
    directory = reference_directory()
    if directory is None:  # pragma: no cover - guarded by the marker
        pytest.skip("no reference tabulations available")
    path = directory / "attenuations_MilkyWay_dustHeightRatio1.0.hdf5"
    if not path.exists():  # pragma: no cover - depends on what is installed
        pytest.skip(f"{path} is not present")
    return path
