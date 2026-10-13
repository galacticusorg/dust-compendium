# Running a campaign

A *campaign* is one configuration file and the thousands of radiative transfer
models it expands into. Five commands take it from the file to a tabulation.

## Describing what to compute

Lengths are in units of the disk scale length. The absolute scale cancels:
optical depths and wavelengths fix everything that matters.

```yaml
label: compendium:exp:sech:Hernquist:dustD03Rv3.1
dust:
  file: hyperion-dust-0.1.0/dust_files/d03_3.1_6.0_A.hdf5
geometry:
  cutOff: 10.0
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust:    {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137,
                opticalDepth: {minimum: 0.01, maximum: 1.0e4, count: 60, includeZero: true}}
    spheroid:
      stellar: {profile: hernquist, truncation: 10.0,
                scaleRadial: {minimum: 1.0e-3, maximum: 1.0e2, count: 51}}
tabulation:
  wavelengths:  {minimum: 0.01, maximum: 3.0, count: 250}
  inclinations: {minimum: 0.0, maximum: 90.0, count: 46, spacing: linear}
  photons: 100000
```

Any value written as a range, or as a list, becomes an axis of the tabulation.
Anything written as a number is held fixed.

## Seeing what it would run, before running it

```console
$ dust-compendium validate configs/compendium-d03-rv3.1.yaml
  axes         :
    opticalDepth:disk                        61 values  from 0 to 10000
    scaleRadial:spheroid                     51 values  from 0.001 to 100
  emitters     :
    disk                                     61 models  over opticalDepth:disk
    spheroid                               3111 models  over opticalDepth:disk, scaleRadial:spheroid
  total        : 3172 models
```

This needs no Hyperion, and is the quickest way to find out that a campaign is
ten times the size you meant before committing a cluster to it.

Note that the disk is tabulated against optical depth alone. An axis which
cannot change a component's light — here, the size of a spheroid holding no dust
— is not tabulated against, since doing so would run the same model fifty-one
times.

## Building, running, collecting

```bash
dust-compendium build   config.yaml -o models
dust-compendium run     config.yaml -m models -o output --concurrency 8
dust-compendium collect config.yaml -o output -t attenuations.hdf5
```

`build` writes one Hyperion input per model. `run` solves them. `collect`
divides each by the model with no dust, fits the high optical depth
extrapolation, and writes the tabulation.

Each step skips work already done, so an interrupted campaign is resumed by
running the same command again. `run` checks that a solved model actually
contains the SEDs it was meant to produce rather than trusting the solver's exit
status — Hyperion aborts on some conditions and still exits zero.

## On a cluster

```bash
dust-compendium run config.yaml -m models -o output \
    --scheduler slurm --concurrency 40 \
    --nodes 4 --tasks 64 --partition compute --walltime 02:00:00
```

The Slurm backend writes a batch script per model, submits with `sbatch`, and
watches `squeue` and `sacct`. Only this campaign's jobs count towards
`--concurrency`, so a busy account cannot stall it, and a transient scheduler
failure is never mistaken for jobs having finished.

`--tasks` matters more than it looks. The solver's cost rises steeply with
optical depth, and for the published compendium's grid the most opaque models
need about a week serially -- past most queue limits. Sixteen tasks brings that
to around ten hours. The core-hours are the same either way; only the wall time
moves. It needs the MPI solver (`hyperion_cyl_mpi`) built.

The solver has to be started by the `mpirun` of the MPI installation it was
linked against. Any other one -- including an MPICH which a conda environment
brought along, and which may be first on `PATH` -- starts each task as a
separate single-rank job, so that sixteen copies of the solver each run the
whole model and race to write one output. `run` therefore takes the `mpirun`
from beside the MPI library the solver loads, falling back to `PATH` only when
there is none, and prints the launcher it chose. `--launcher` overrides it, for
a site which insists on `srun` for instance: `--launcher 'srun --mpi=pmix -n'`.
The task count is appended to whatever is given. A model whose log shows more
than one copy of the solver started is reported as failed, and its output
removed so that the next run solves it again.

## Re-running the published compendium

`configs/compendium-*.yaml` cover the five published grain models, 3,172 models
each. They differ from the originals in using 23 inclinations spaced uniformly
in their cosine rather than 46 spaced uniformly in angle, and `nested` walls
with `average` sampling rather than the published grid -- see
[Published datasets](datasets.md) for why the latter is deliberate.

```bash
# Once: the grain files, which nothing else can produce.
pip download --no-binary :all: --no-deps hyperion-dust
tar xzf hyperion-dust-0.1.0.tar.gz
dust-compendium grains hyperion-dust-0.1.0

for grains in d03-rv3.1 d03-rv4.0 d03-rv5.5 kmh94-hg kmh94-full; do
    config=configs/compendium-$grains.yaml
    dust-compendium build   $config -o models/$grains
    dust-compendium run     $config -m models/$grains -o output/$grains \
        --scheduler slurm --concurrency 100 --tasks 16 \
        --partition compute --walltime 24:00:00
    dust-compendium collect $config -o output/$grains \
        --tabulation compendium-$grains.hdf5
done
```

Keep the three directories -- the dust files, the models and the outputs --
together, since the links between them are relative to each other. Moving or
copying the whole tree is fine, and was tested; moving one part out of it is
not. The inputs have to survive until `collect` has run.

Each step can be interrupted and run again; it picks up what the one before it
left. Budget around 5.8 GB of disk and, at 250 wavelengths and 100,000 photons,
of order 0.8 million core-hours for all five -- so measure before committing,
starting with whether cost really is linear in the number of wavelengths
(`configs/calibration-25lambda.yaml`), which is assumed and untested.

## Looking at the result

```bash
dust-compendium plot attenuations.hdf5 -e disk -q extinction -a inclination \
    --colour-by opticalDepth -o extinction.pdf
dust-compendium plot attenuations.hdf5 -e spheroid -q curve -c spheroidScaleRadial
dust-compendium compare attenuations.hdf5 --atlas /path/to/atlasFerrara2000/....hdf5
```

`plot` draws any quantity — extinction at any wavelength, or the reddening
{math}`R_\mathrm{V}` — against any axis, optionally as a family of curves.
`compare` measures agreement against a published atlas rather than merely
drawing the two together.
