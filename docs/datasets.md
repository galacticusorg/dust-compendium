# Published datasets

The tabulations published on Zenodo, and listed in the
[Galacticus documentation](https://galacticus.readthedocs.io/en/latest/manuals/user-guide/data/dust-compendium-datasets.html),
were produced by the original scripts. Rewriting those scripts, and checking the
result against them and against Ferrara et al. (1999), turned up several defects.
They are recorded here because they affect data which is published and in use.

Nothing here has been changed on Zenodo. What to do about these is not this
package's decision.

## Reaching published data

### The spheroid emission map, in the compendium tabulations

The antiderivative used for a spheroid's cell masses is odd in height, but the
`R = 0`, `|z| = r_s` corner was assigned a constant without regard to the sign
of {math}`z`. Below the midplane the sign is wrong.

That corner is reached whenever `z = -r_s` falls on a vertical wall, which for
walls spanning {math}`\pm c\,r_\mathrm{s}` in {math}`n` cells happens exactly
when {math}`(n/2)(c-1)/c` is a whole number. With a hundred cells that holds for
the compendium's cut off of ten and fails for the Ferrara-matched models' cut
off of six.

For a Hernquist spheroid with a scale radius equal to the disk scale length, the
resulting emission map puts 54% of all the light into the two cells either side
of that wall, which between them hold 0.0003% of the mass, and 77% of the
emission below the midplane rather than half.

**Affected: the `compendium_*` tabulations, for every spheroid scale radius at
or above 0.137.** The Ferrara-matched ones escape it, which their agreement with
Ferrara's own atlas — a median of 0.012 in transmission — bears out.

### The inclination axis, in the Ferrara-matched tabulations

The driver passed each model's own inclinations to the model builder, so the
models were solved at them, and then built the axis it recorded as a uniform
grid from zero to ninety regardless. The published files hold attenuations
computed at 9.3, 22.9, 30 ... degrees under an inclination axis reading
0, 11.25, 22.5 ..., so anything interpolating them reads the wrong angle.

Which grid the data actually sits at is measurable: index for index against
Ferrara's angles the published file agrees with their atlas to a median of 0.003
in transmission, while interpolating the atlas onto the uniform axis is nearly
three times worse.

**Affected: the `Ferrara1999_*` original-resolution tabulations.**

## Latent, and not reaching published data

These were found in the same code. The published grids happen to avoid all of
them, but each would have produced silently wrong results on a slightly
different grid, and each is now a regression test.

- Cell corners were combined through nested absolute values rather than a plain
  two-dimensional difference, and the disk's vertical factor folded about the
  midplane. A cell straddling the midplane came out wrong, and one symmetric
  about it came out as exactly zero. The published grids use an even number of
  vertical cells, which puts the midplane on a wall.
- The spheroid antiderivatives handled `R = 0` and `R = r_s` but not the rest of
  the sphere `r = r_s`, where a corner is wrong by up to a factor of a hundred —
  and not always visibly, since the result can be finite rather than NaN.
- A small but non-zero radius returned NaN.
- The Hernquist form lost all precision within about {math}`10^{-5}` of
  `R = r_s`.
- `spheroidCutOff` was parsed and never applied, so the Ferrara-matched models
  were not truncated at five effective radii as intended. See below.

## An open question

Ferrara et al. describe truncating their spheroids at five effective radii.
Applying that, at the {math}`r_\mathrm{s} = 1.16 R_\mathrm{e}` relation the
original assumed, makes agreement with their atlas *worse*: measured by running
two campaigns identical but for the truncation, it lowers the spheroid
transmission by a median of 0.009 and up to 0.08, growing with optical depth as
it removes the outer and least attenuated light.

So the atlas behaves as though those spheroids were not truncated there. Either
the relation between scale and effective radius is not the one assumed, or the
truncation is weaker than described. `configs/ferrara1999-milkyway.yaml` leaves
them untruncated, since matching the atlas is what that configuration is for.

## What agreement looks like

Reproducing Ferrara et al. (1999) with matched grains, geometry and grids, and
comparing with their atlas — which is published to two decimal places, so 0.005
in transmission is the floor on any agreement:

| | disk | spheroid, truncated at 5 R_e |
|---|---|---|
| the original scripts | 0.0032 | 0.0121 (never truncated) |
| this package | 0.0034 | 0.0245 |

Median absolute difference in transmission. For the disk, 99.9% of entries agree
to better than 0.02, and this package reproduces the atlas as well as the
original scripts do.

The spheroid column is not like for like: the original never applied a
truncation and the figure above was computed with one. The measured effect of
the truncation, a median of 0.009 in the same direction, accounts for the
difference, which is why the Ferrara-matched configuration no longer applies
one. An untruncated campaign at matched resolution has not been run end to end
here, so no number is quoted for it.
