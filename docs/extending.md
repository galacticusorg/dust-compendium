# Extending

## A new density profile

A profile says what its density is, how much mass sits in a cell, and what
optical depth means for it. Subclass
{class}`~dustcompendium.geometry.Profile`, or
{class}`~dustcompendium.geometry.Spheroid` for a spherical one, where only the
shape function, its radial integral and the column at the scale radius are
needed.

Then register it, so configurations can name it:

```python
SPHEROID_PROFILES["nfw"] = NFWSpheroid
```

Two things are worth doing for a new profile, because they caught real errors in
the profiles already here. Check its cell masses against numerical quadrature,
including where any closed form has a removable singularity — a corner landing
exactly on one is not exotic, since grid walls fall on round numbers. And check
that scaling it to an optical depth reproduces that optical depth, by
integrating the normalized density along the ray it is defined on.

## A new vertical structure for disks

A `VerticalStructure` supplies a shape {math}`f(z)` with {math}`f(0)=1` and its
antiderivative. Define the antiderivative with {math}`F(0)=0`, carrying the sign
of {math}`z`, so that {math}`\int_{z_0}^{z_1} f = F(z_1) - F(z_0)` holds for any
interval. Folding about the midplane instead gives zero for a cell symmetric
about it, which is where most of a disk's mass lies.

## A new scheduler

A scheduler runs a list of jobs and says which of them worked. Subclass
{class}`~dustcompendium.runner.Scheduler`, implement `run`, and register it in
`SCHEDULERS`.

Return a result for every job, in the order given. Do not treat an exit status
of zero as proof of success on its own: Hyperion aborts on some conditions and
still exits zero. The command line checks the output for the SEDs it was meant
to produce, and a scheduler need not, but it must not report a job as having
succeeded when it never ran.

## A new output format version

Version 2 is a strict superset of version 1. If a further version is needed,
keep that property: the names version 1 uses should go on meaning what they
mean, so that an old reader either works or fails loudly.

Failing loudly is the design. Galacticus checks the shape of every array against
the axes, which is what makes a file it cannot understand an error rather than
quietly wrong attenuation.
