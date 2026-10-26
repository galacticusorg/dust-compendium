# Changelog

## 0.2.0

The release the compendium re-run was computed with. Everything in
[the published version 2 tabulations](https://doi.org/10.5281/zenodo.6335020)
came from this code.

### Added

- `grains` builds the five standard Draine (2003) and Kim, Martin & Hendry
  (1994) dust files from an unpacked `hyperion-dust` source tree. Nothing else
  could: that package converts its tabulations with a `setup.py build_dust`
  written against distutils, which Python 3.12 removed, so the files the
  compendium configurations name had no working build at all. Checked against
  the file used for the 2018 paper, the regenerated V-band opacity agrees to
  3e-5 — the residual being that the original rounded the speed of light.
- `spacing: cosine` on a range, which spaces inclinations uniformly in their
  cosine. Measured against a finer grid, 23 such points interpolate as
  accurately as 46 uniform in angle, for half the cost, and are the grid
  randomly oriented galaxies are uniform on.
- Hyperion's modified random walk, through `mrw` and `mrwGamma`. It is plumbed,
  tested and off by default; measured, it never engages with
  `n_initial_iterations = 0`, since its trigger is the temperature-dependent
  Rosseland mean free path.
- Configurations for the five published grain models, for sizing a campaign,
  for the random walk, and for measuring the cost of peel-off directions.

### Changed

- Grain properties are linked into a model rather than copied, and the solver
  is told to link its output to its input rather than copy that. A model goes
  from 13.0 MB of input and 13.0 MB of output to 0.18 and 0.18; for the
  15,860-model re-run, from about 410 GB to 5.8. The links are relative, so a
  solved tree can be moved or copied anywhere as long as it travels together.
  `--embed-dust` restores self-contained models.

### Fixed

- The high optical depth extrapolation no longer writes not-a-number. Where
  transmission underflows to zero before the largest tabulated depth — a
  compact spheroid buried in disk dust — the cell is recorded as opaque
  instead. Galacticus extrapolates by default and has no NaN guard, so what
  was written before would have become NaN attenuations.
- A fit which rises is held flat at the tabulation boundary. Transmission
  cannot increase with optical depth, but a least-squares fit to a barely
  attenuated cell can come out with a gradient that is positive noise, and
  extrapolated far enough that reports more light than the galaxy emitted.
- `run` launches the MPI solver with the `mpirun` beside the MPI library the
  solver actually loads, found with `ldd`, rather than whichever came first on
  `PATH`. A mismatch started every task as its own single-rank job, all solving
  the whole model and racing to write the same output. A log showing more than
  one `Started on` line now fails the model and removes its output, so the next
  run does not skip it as solved. `--launcher` overrides, for `srun`.
- That lookup now happens once per run rather than once per model, which had
  held a 3,172-model campaign up for the better part of an hour before it
  submitted anything.
- `collect` records which axis the extrapolation was fitted along, so a
  campaign with no disk dust can be assembled.
- `collect` no longer crashes when too few optical depths fall in the fitting
  range; it warns and leaves the coefficients absent.
- `grains` checks its arguments before requiring Hyperion, so a wrong path or
  an unknown grain model is reported as what it is, and defaults its output
  into the source tree it was given rather than the working directory.
- A missing Hyperion now names the interpreter that was searched, which is
  usually the whole answer.

## 0.1.0

First release as a package. Replaces the Perl and Python 2 scripts used for
[Benson (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B), which
are preserved verbatim in the first commit of this repository.

### What it does

- Galaxies are a set of named components, each of which may carry a stellar
  profile, a dust profile, both or neither, so dust in the spheroid as well as
  the disk is a change to a configuration rather than to any code.
- Campaigns are described in YAML and validated on load. Any value written as a
  range becomes an axis of the tabulation. `validate` reports what a campaign
  would run without needing Hyperion.
- `build`, `run` and `collect` each resume from what the last left on disk.
  `run` has local and Slurm backends; PBS is not carried over.
- The output format is versioned, self-describing, and reproduces the layout
  Galacticus reads.
- `plot` replaces nine near-identical plotting scripts, and drops the gnuplot,
  cairolatex and pdflatex chain. `compare` measures agreement against a
  published atlas.
- The core — geometry, optical depths, the reader and writer, post-processing,
  plotting and comparison — needs no Hyperion at all.

### Fixed, relative to the original scripts

Six defects, found by testing the ported code against numerical quadrature and
against Ferrara et al. (1999). Two of them reached published data; see the
documentation for which tabulations are affected.

- The spheroid cell-mass antiderivative carried the wrong sign below the
  midplane at the `R = 0`, `|z| = r_s` corner, which the compendium grids reach.
  For a spheroid the size of the disk this put 54% of its light into two cells
  holding 0.0003% of its mass.
- The recorded inclination axis was always a uniform grid, even when the models
  were solved at other angles, which the Ferrara-matched tabulations were.
- Cell corners were combined through nested absolute values, so a cell
  straddling the midplane was wrong and one symmetric about it came out empty.
- The spheroid antiderivatives were unhandled on the sphere `r = r_s`, returned
  NaN at small non-zero radius, and lost all precision near `R = r_s`.
- `spheroidCutOff` was parsed and never applied; spheroids can now be truncated.
- The source spectrum spanned exactly the dust's frequency range, so a photon
  drawn at its edge aborted the solve. Hyperion exits zero when it aborts, so
  those models were recorded as successes; solved models are now checked for the
  SEDs they were meant to produce.

### Added beyond the original

- `spacing: nested`, which resolves a thin disk inside a large spheroid. The
  published grid cannot: at a spheroid ten times the disk scale length it leaves
  a model asked for an optical depth of one with one of zero.
- `sampling: average`, which puts the profile's true mass in each cell. Point
  sampling loses 7% of a Jaffe dust profile, and up to 95% in the innermost cell.
- The extrapolation fit's residual is stored beside its coefficients, so the
  power-law assumption can be checked rather than trusted.
- Uncertainties propagate from both the attenuated model and the reference.
