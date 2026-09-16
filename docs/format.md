# The file format

A tabulation is an HDF5 file. The layout is a contract:
[Galacticus](https://github.com/galacticusorg/galacticus) reads these files in
`source/dust/attenuation/atlas_compendium.F90`, which checks the shape of every
array it reads against the axes — "a transposed read would otherwise show up
much later as quietly wrong attenuation".

Fortran indexes in the reverse of C, so an array written as
`[wavelength, inclination, opticalDepth]` is read as
`[opticalDepth, inclination, wavelength]`. The tests state the shapes in both
orders.

## Version 1

The published layout: a single optical depth, belonging to the disk, and a
spheroid tabulated against its size.

```
attribute  opacity                                   recognises the file
attribute  formatVersion, label, timeStamp, ...
dataset    wavelength                        [nlam]  microns
dataset    inclination                       [ninc]  degrees, 0 face-on
dataset    opticalDepth                      [ntau]  may include an exact zero
dataset    spheroidScaleRadial               [nrs]   in disk scale lengths
dataset    attenuationDisk                   [lam, inc, tau]
dataset    attenuationSpheroid               [lam, inc, tau, rs]
dataset    attenuationUncertaintyDisk        [lam, inc, tau]
dataset    attenuationUncertaintySpheroid    [lam, inc, tau, rs]
dataset    extrapolationCoefficientsDisk     [2, lam, inc]
dataset    extrapolationCoefficientsSpheroid [2, lam, inc, rs]
dataset    extrapolationResidualDisk         [lam, inc]
dataset    extrapolationResidualSpheroid     [lam, inc, rs]
```

Galacticus does not read the uncertainties. They are half of every published
file, and they are what says whether a tabulation has converged.

The extrapolation coefficients are the {math}`c_0` and {math}`c_1` of
{math}`\exp(c_0 + c_1\ln\tau)`, in that order along the first axis.

## Version 2

Written when a campaign has more axes than that — dust in the spheroid as well
as the disk, or any other component. It is a strict superset: every version 1
name keeps its meaning, and new axes are added.

```
dataset    spheroidOpticalDepth              [ntaus]
dataset    attenuationDisk                   [lam, inc, tau, taus]
dataset    attenuationSpheroid               [lam, inc, tau, taus, rs]
dataset    extrapolationCoefficientsDisk     [2, lam, inc, taus]
```

A version 2 file is correctly *rejected* by the current Fortran class, through
the same shape checks, rather than quietly misread. Reading one needs a
companion class in Galacticus, which does not yet exist.

## Self-description

Two things the published files did not record, and these do.

Each attenuation dataset carries an `axes` attribute naming its dimensions, so a
reader need not hard-code the rank.

Each axis dataset carries an `axis` attribute giving the axis it came from —
`opticalDepth` is the disk's, `spheroidScaleRadial` the spheroid's size, and
those names are not a pattern and do not invert.

Each component's profile is recorded as an attribute. Galacticus keeps a
hard-coded table of twenty-one published file names paired with their spheroid
profiles precisely because the files never said which they were computed for.
