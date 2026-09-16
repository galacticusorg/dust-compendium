# Installation

```bash
pip install dust-compendium
```

That gives the library and the command line tool: the geometry, the tabulation
reader and writer, post-processing, plotting and comparison. Reading and
analysing an existing tabulation needs nothing further.

Computing *new* tabulations needs Hyperion, which comes in two parts and is not
a dependency of this package.

## Hyperion's Python package

The current release on PyPI, 0.9.11 of April 2024, calls `np.string_` when
writing a model. NumPy 2.0 removed it, so that release cannot build models on
NumPy 2 or later. Upstream has fixed this but has not released since, so for now
install from source:

```bash
git clone https://github.com/hyperion-rt/hyperion.git
pip install ./hyperion
```

There is deliberately no `hyperion` extra on this package: no version that `pip`
can resolve works, and an extra that always fails would be worse than saying so
here.

## Hyperion's solvers

The radiative transfer itself is done by Fortran programs — `hyperion_cyl` for
the cylindrical grids used here, and `hyperion_cyl_mpi` for the parallel
version. They are in neither the wheel nor the sdist install, and must be built
from the same source tree:

```bash
cd hyperion
./configure --prefix=$PREFIX
make
make install
```

`configure` finds HDF5 through `h5fc` and MPI through `mpif90`, so both need to
be on `PATH` before running it. Put `$PREFIX/bin` on `PATH` afterwards;
`dust-compendium run` looks for the solver there and says so if it cannot find
one.

## Which step needs what

| step | Hyperion's Python package | a solver binary |
|---|---|---|
| `validate` | no | no |
| `build` | yes | no |
| `run` | no | yes |
| `collect` | yes | no |
| `plot`, `compare` | no | no |

`run` needs neither the Python package nor a dust file because everything
physical is already written into the model inputs. A cluster therefore needs a
much lighter installation than the machine that prepared the campaign.

## For development

```bash
pip install -e ".[dev,plots,docs]"
pre-commit run --all-files
pytest
```

Tests needing Hyperion, a solver binary, or published reference tabulations are
marked `hyperion`, `solver` and `reference`, and skip when those are absent. The
rest — the great majority — run anywhere.
