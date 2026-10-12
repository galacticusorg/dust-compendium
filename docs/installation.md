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

## On a cluster

A real campaign is tens of thousands of models and hundreds of gigabytes of
intermediate output, so it wants a cluster. The work splits naturally: prepare
the campaign where it is convenient, solve it where the processors are.

```bash
# wherever the campaign is being prepared
dust-compendium build config.yaml -o models
rsync -a models/ cluster:~/campaign/models/

# on the cluster
dust-compendium run     config.yaml -m models -o output --scheduler slurm ...
dust-compendium collect config.yaml -o output -t attenuations.hdf5
```

Collecting on the cluster is usually the right way round: the solved models are
much larger than the tabulation they reduce to, and copying them back to collect
elsewhere moves far more data than it needs to.

### An environment with conda

```bash
conda create -n dust-compendium -c conda-forge \
    python=3.12 gfortran hdf5 mpich pip
conda activate dust-compendium
which h5fc mpif90          # hyperion's configure needs both
```

HDF5 brings the Fortran wrappers that `configure` looks for. MPI is needed only
to satisfy `configure`: with tens of thousands of independent models, one serial
solver per job is simpler and more efficient than splitting single models across
ranks, so the MPI solver is never built or run and conda's MPI never competes
with the site's.

```bash
pip install dust-compendium

git clone https://github.com/hyperion-rt/hyperion.git
cd hyperion
git submodule update --init --recursive
pip install .                      # needed by `collect`, not by `run`

./configure --prefix=$CONDA_PREFIX
make cyl_serial                    # these models are axisymmetric
make install
hyperion_cyl                       # prints its usage line
```

Build the solvers from the same checkout the Python package was installed from.
That package writes the model inputs and the solver reads them, and keeping both
at one version removes a class of confusing failure. Installing into
`$CONDA_PREFIX` means activating the environment is enough to put the solver on
`PATH` with its libraries alongside.

conda-forge also carries a `hyperion-fortran` package, which would supply the
solvers without building anything. It is worth a try, but it is the released
0.9.11 while the Python package has to be newer than that, and a model input
written by one version is not guaranteed to be read by another. A mismatch shows
up immediately: the solver either refuses the input or writes an output with an
empty peeled group, which `run` reports rather than passing over.

### In a job script

```bash
source $(conda info --base)/etc/profile.d/conda.sh
conda activate dust-compendium
```

Activate in the script itself, not only in a login shell. Otherwise jobs fail
while interactive runs succeed, which is a tiresome thing to debug.

### Sizing a campaign first

`configs/calibration.yaml` is twelve models at production settings, spanning the
optical depth range, for finding out what a campaign will cost on a machine that
has not run one. Run it with `--limit 4` first, which takes the cheaper models
and serves as a smoke test of the scheduler before anything expensive is
submitted to it. Each solver writes its processor time at the end of its log.

## For development

```bash
pip install -e ".[dev,plots,docs]"
pre-commit run --all-files
pytest
```

Tests needing Hyperion, a solver binary, or published reference tabulations are
marked `hyperion`, `solver` and `reference`, and skip when those are absent. The
rest — the great majority — run anywhere.
