# Dust Compendium

Dust attenuation tables for simple galactic geometries, computed by running the
[Hyperion](http://www.hyperion-rt.org) Monte Carlo radiative transfer code over
a grid of galaxy models.

The tables give the fraction of starlight escaping a galaxy as a function of
wavelength, inclination, dust optical depth, and the size of the spheroid
relative to the disk. They are consumed by
[Galacticus](https://github.com/galacticusorg/galacticus) through its
`dustAttenuationAtlasCompendium` class. The method is described in
[Benson (2018)](https://ui.adsabs.harvard.edu/abs/2018RNAAS...2..188B).

Published tabulations are listed in the
[Galacticus documentation](https://galacticus.readthedocs.io/en/latest/manuals/user-guide/data/dust-compendium-datasets.html)
and archived on Zenodo.

## Status

Being restructured from the original scripts used for the 2018 paper into an
installable Python package. The original Perl and Python 2 scripts are preserved
in the first commit of this repository.

## Installation

```bash
pip install dust-compendium
```

That gives the library: the geometry and optical depth machinery, the tabulation
reader and writer, post-processing, and plotting. It is deliberately usable
without a radiative transfer install, so that reading and analyzing a published
tabulation needs nothing further.

Computing new tabulations additionally needs Hyperion, in two parts:

1. **The Python front-end.** The current release on PyPI (0.9.11) calls
   `np.string_` when writing a model, which NumPy 2.0 removed, so it cannot
   build models on NumPy >= 2. Upstream master has fixed this but is unreleased,
   so for now install from source:

   ```bash
   git clone https://github.com/hyperion-rt/hyperion.git
   pip install ./hyperion
   ```

2. **The Fortran solvers** (`hyperion_cyl`, `hyperion_cyl_mpi`), which do the
   radiative transfer. These are in neither the wheel nor a `pip install`, and
   must be built from the same source tree, against HDF5 with Fortran bindings
   and (for the MPI solvers) an MPI implementation:

   ```bash
   cd hyperion && ./configure --prefix=$PREFIX && make && make install
   ```

   `configure` locates HDF5 through `h5fc` and MPI through `mpif90`, so both
   need to be on `PATH`.

## License

MIT. See [LICENSE](LICENSE).
