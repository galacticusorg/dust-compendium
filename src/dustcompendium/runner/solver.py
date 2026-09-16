r"""Locating and invoking the Hyperion solver binaries.

These are Fortran programs which ``pip`` does not install -- neither the wheel
nor the sdist carries them -- so they have to be built from the Hyperion source
and put on ``PATH``. See the installation notes in the README.
"""

import shutil
from collections.abc import Sequence
from pathlib import Path

__all__ = ["SOLVERS", "solver_command", "which_solver"]

#: Solver binaries by grid geometry, serial and MPI. The models here are
#: cylindrical, so ``cylindrical`` is the one that matters.
SOLVERS = {
    "cylindrical": ("hyperion_cyl", "hyperion_cyl_mpi"),
    "cartesian": ("hyperion_car", "hyperion_car_mpi"),
    "spherical": ("hyperion_sph", "hyperion_sph_mpi"),
}


def which_solver(geometry: str = "cylindrical", parallel: bool = False) -> str:
    """Find a solver binary on ``PATH``.

    Raises
    ------
    KeyError
        If the geometry is not one Hyperion provides.
    FileNotFoundError
        If the binary is not on ``PATH``, with a pointer to the build notes.
    """
    try:
        serial, mpi = SOLVERS[geometry]
    except KeyError:
        known = ", ".join(sorted(SOLVERS))
        raise KeyError(f"unknown geometry {geometry!r}; known geometries are {known}") from None
    name = mpi if parallel else serial
    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name} is not on PATH. The Hyperion solvers are Fortran binaries which pip "
            "does not install; build them from the Hyperion source and add their bin "
            "directory to PATH. See the installation notes in the README."
        )
    return found


def solver_command(
    source: Path,
    result: Path,
    geometry: str = "cylindrical",
    tasks: int = 1,
    launcher: Sequence[str] = ("mpirun", "-np"),
    overwrite: bool = True,
) -> list[str]:
    """The command which solves one model.

    Parameters
    ----------
    source
        The model input file, as written by
        :func:`~dustcompendium.hyperion_model.write_model`.
    result
        Where the solved model should go.
    geometry
        Which solver to use; the models here are ``cylindrical``.
    tasks
        Number of MPI tasks. One uses the serial solver and no launcher.
    launcher
        How to start an MPI job, as a prefix taking the task count.
    overwrite
        Pass the solver's ``-f``, so that an existing output is replaced rather
        than the run refusing.
    """
    if tasks < 1:
        raise ValueError(f"tasks must be at least one, got {tasks}")
    binary = which_solver(geometry, parallel=tasks > 1)
    prefix = [*launcher, str(tasks)] if tasks > 1 else []
    flags = ["-f"] if overwrite else []
    return [*prefix, binary, *flags, str(source), str(result)]
