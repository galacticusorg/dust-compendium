"""Command line interface."""

import typer

from . import __version__

app = typer.Typer(
    help="Compute and inspect dust attenuation tables for simple galactic geometries.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Compute and inspect dust attenuation tables for simple galactic geometries."""


@app.command()
def version() -> None:
    """Report the installed version."""
    typer.echo(__version__)
