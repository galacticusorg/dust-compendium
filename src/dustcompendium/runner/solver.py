r"""Locating and invoking the Hyperion solver binaries.

These are Fortran programs which ``pip`` does not install -- neither the wheel
nor the sdist carries them -- so they have to be built from the Hyperion source
and put on ``PATH``. See the installation notes in the README.
"""

import os
import re
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

__all__ = [
    "SOLVERS",
    "is_solved",
    "relink_input",
    "solver_command",
    "solver_copies",
    "which_launcher",
    "which_solver",
]

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


def which_launcher(binary: str) -> list[str]:
    """The MPI launcher which matches a solver binary, as a prefix taking the task count.

    An MPI program has to be started by the launcher of the MPI library it was
    linked against. Any other launcher -- another implementation, or even another
    installation of the same one -- starts each process as a separate,
    single-rank job, so that ``mpirun -np 16`` runs sixteen independent copies of
    the solver, all writing the same output. Whichever ``mpirun`` happens to be
    first on ``PATH`` is no guide: a conda environment holding the Python side,
    for one, can bring its own MPICH. So the launcher is looked for beside the
    MPI library the binary loads, falling back to ``PATH`` only when that cannot
    be found.

    Raises
    ------
    FileNotFoundError
        If no launcher can be found at all.
    """
    try:
        linked = subprocess.run(["ldd", binary], capture_output=True, text=True, check=False).stdout
    except OSError:
        linked = ""
    match = re.search(r"libmpi\.so\S*\s+=>\s+(\S+)", linked)
    if match:
        candidate = Path(match.group(1)).parent.parent / "bin" / "mpirun"
        if os.access(candidate, os.X_OK):
            return [str(candidate), "-np"]
    found = shutil.which("mpirun")
    if found is None:
        raise FileNotFoundError(
            f"no MPI launcher could be found for {binary}: there is no mpirun beside the MPI "
            "library it loads, nor on PATH. Name one with --launcher."
        )
    return [str(Path(found).absolute()), "-np"]


def solver_command(
    source: Path,
    result: Path,
    geometry: str = "cylindrical",
    tasks: int = 1,
    launcher: Sequence[str] | None = None,
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
        How to start an MPI job, as a prefix taking the task count. By default,
        the one matching the solver's MPI library; see :func:`which_launcher`.
    overwrite
        Pass the solver's ``-f``, so that an existing output is replaced rather
        than the run refusing.
    """
    if tasks < 1:
        raise ValueError(f"tasks must be at least one, got {tasks}")
    binary = which_solver(geometry, parallel=tasks > 1)
    prefix = []
    if tasks > 1:
        prefix = [*(which_launcher(binary) if launcher is None else launcher), str(tasks)]
    flags = ["-f"] if overwrite else []
    return [*prefix, binary, *flags, str(source), str(result)]


def relink_input(result: Path, source: Path) -> bool:
    """Make a solved model's link to its input relative rather than absolute.

    When the input is linked instead of copied, the solver records the path it
    was handed, which is absolute -- so copying the results anywhere, to collect
    them on another machine for instance, leaves every output pointing at a
    directory which no longer exists. Rewriting the link relative to the output
    makes the whole tree movable, which is the only form the chain through to
    the dust file is usable in anyway.

    Returns whether a link was rewritten; a copied input has none.
    """
    import h5py

    try:
        with h5py.File(result, "r+") as handle:
            link = handle.get("Input", getlink=True)
            if not isinstance(link, h5py.ExternalLink):
                return False
            relative = os.path.relpath(source.resolve(), result.resolve().parent)
            if link.filename == relative:
                return False
            target = link.path
            del handle["Input"]
            handle["Input"] = h5py.ExternalLink(relative, target)
    except OSError:
        return False
    return True


def is_solved(path: Path) -> bool:
    """Whether a solved model actually holds the SEDs it was meant to produce.

    An exit status of zero is not enough. Hyperion aborts on some conditions --
    a photon at a frequency its dust is not defined at, for one -- by printing
    an error and stopping, and *still exits zero*, leaving an output file whose
    peeled group is empty. Taking the status at face value would record such a
    model as solved, skip it on every later run, and only surface it as a gap
    when the tabulation was assembled.
    """
    import h5py

    if not path.exists():
        return False
    try:
        with h5py.File(path, "r") as handle:
            groups = handle.get("Peeled")
            return bool(groups is not None and len(groups) > 0)
    except OSError:
        # A truncated or unreadable file is not a solved one.
        return False


def solver_copies(log: Path) -> int:
    """How many copies of the solver wrote to a log.

    Hyperion prints its banner from the first rank only, so a correctly launched
    MPI run has exactly one ``Started on`` line however many tasks it has. More
    than one means the launcher did not match the solver's MPI library and each
    process ran the whole model by itself (see :func:`which_launcher`). Such a
    run is not a solve whatever its exit status: the copies race to write the
    same output, and one which happens to finish last can leave a file which
    looks complete.

    A missing log counts as none.
    """
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return len(re.findall(r"^\s*Started on ", text, flags=re.MULTILINE))
