"""Command line interface."""

from pathlib import Path
from typing import Annotated

import typer

from . import __version__

app = typer.Typer(
    help="Compute and inspect dust attenuation tables for simple galactic geometries.",
    no_args_is_help=True,
)

ConfigArgument = Annotated[
    Path, typer.Argument(exists=True, dir_okay=False, help="A campaign configuration, in YAML.")
]


@app.callback()
def main() -> None:
    """Compute and inspect dust attenuation tables for simple galactic geometries."""


@app.command()
def version() -> None:
    """Report the installed version."""
    typer.echo(__version__)


@app.command()
def validate(config: ConfigArgument) -> None:
    """Check a campaign configuration and describe what it would run.

    Reads the configuration, expands every range into an axis, and reports the
    models that would result. Needs no Hyperion, so it is the quickest way to
    see whether a campaign is the size you expected before committing a cluster
    to it.
    """
    from .campaign import Campaign
    from .config import load_campaign

    campaign = Campaign(load_campaign(str(config)))
    settings = campaign.config
    typer.echo(f"{settings.label}")
    if settings.description:
        typer.echo(f"  {settings.description}")
    grains = settings.dust.file or f"Ferrara/Gordon {settings.dust.ferrara}"
    typer.echo(f"  grains       : {grains}")
    typer.echo(
        f"  grid         : {settings.geometry.radial_cells}"
        f" x {settings.geometry.vertical_cells} cells,"
        f" {settings.geometry.spacing} spacing,"
        f" {settings.geometry.sampling} sampling,"
        f" cut off {settings.geometry.cut_off}"
    )
    typer.echo(
        f"  tabulated at : {campaign.wavelengths.size} wavelengths"
        f" from {campaign.wavelengths.min():g} to {campaign.wavelengths.max():g} micron,"
        f" {campaign.inclinations.size} inclinations"
    )
    typer.echo("  axes         :")
    for axis in campaign.axes:
        typer.echo(
            f"    {axis.name:<36} {len(axis):>5} values  from {axis.values.min():g} to {axis.values.max():g}"
        )
    typer.echo("  emitters     :")
    for emitter in campaign.emitters:
        shape = campaign.shape_for(emitter)
        names = ", ".join(axis.name for axis in campaign.axes_for(emitter)) or "none"
        count = 1
        for length in shape:
            count *= length
        typer.echo(f"    {emitter:<36} {count:>5} models  over {names}")
    typer.echo(f"  total        : {len(campaign)} models")


@app.command()
def build(
    config: ConfigArgument,
    output: Annotated[
        Path, typer.Option("--output", "-o", help="Directory to write model input files into.")
    ] = Path("models"),
    limit: Annotated[int, typer.Option(help="Write at most this many models; 0 writes all of them.")] = 0,
    overwrite: Annotated[bool, typer.Option(help="Rewrite inputs which already exist.")] = False,
) -> None:
    """Write the Hyperion input files for a campaign.

    Needs Hyperion, and the dust file or grain type named in the configuration.
    Existing inputs are left alone unless ``--overwrite`` is given, so an
    interrupted build can be resumed by running it again.
    """
    from .campaign import Campaign
    from .config import load_campaign
    from .dust import ferrara, load_dust
    from .hyperion_model import write_model

    campaign = Campaign(load_campaign(str(config)))
    settings = campaign.config.dust
    if settings.file is not None:
        grains = load_dust(settings.file)
    else:
        assert settings.ferrara is not None  # guaranteed by the configuration
        grains = ferrara.build(settings.ferrara, settings.reference_opacity)

    output.mkdir(parents=True, exist_ok=True)
    options = dict(campaign.grid_options())
    written = skipped = 0
    for run in campaign.runs():
        if limit and written >= limit:
            break
        path = output / f"{run.file_stem}.hdf5"
        if path.exists() and not overwrite:
            skipped += 1
            continue
        write_model(
            run.spec,
            grains,
            str(path),
            sampling=campaign.config.geometry.sampling,
            **options,
        )
        written += 1
    typer.echo(f"wrote {written} model inputs to {output}" + (f", skipped {skipped}" if skipped else ""))
