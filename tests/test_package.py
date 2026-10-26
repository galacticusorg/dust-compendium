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


def test_the_two_recorded_versions_agree():
    """`pyproject.toml` and `__init__.py` each carry the version separately.

    Nothing derives one from the other, so they can drift: the package would
    then report one number while the wheel on PyPI carried another, and the
    mismatch would only show up after a release, which cannot be taken back.
    """
    import tomllib
    from pathlib import Path

    import dustcompendium

    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    declared = tomllib.load(pyproject.open("rb"))["project"]["version"]
    assert declared == dustcompendium.__version__
