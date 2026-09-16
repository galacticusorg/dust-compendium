"""Checks on the package skeleton itself."""

import dustcompendium


def test_version_is_exposed() -> None:
    assert dustcompendium.__version__


def test_cli_reports_version() -> None:
    from typer.testing import CliRunner

    from dustcompendium.cli import app

    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert dustcompendium.__version__ in result.stdout
