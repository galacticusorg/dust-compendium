"""Shared fixtures, and skipping for the optional Hyperion dependency."""

import shutil

import pytest


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

    skip_hyperion = pytest.mark.skip(reason="the Hyperion Python package is not installed")
    skip_solver = pytest.mark.skip(reason="no hyperion_cyl solver binary on PATH")
    for item in items:
        if "hyperion" in item.keywords and not has_hyperion:
            item.add_marker(skip_hyperion)
        if "solver" in item.keywords and not (has_solver and has_hyperion):
            item.add_marker(skip_solver)


@pytest.fixture
def solver_path():
    """Path to the serial cylindrical solver."""
    path = shutil.which("hyperion_cyl")
    if path is None:  # pragma: no cover - guarded by the marker
        pytest.skip("no hyperion_cyl solver binary on PATH")
    return path
