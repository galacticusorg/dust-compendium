"""Tests for the command line interface."""

import pytest
from typer.testing import CliRunner

from dustcompendium.cli import app

TINY = """
label: tiny
dust: {ferrara: milkyWay, referenceOpacity: 250.0}
geometry:
  cutOff: 10.0
  radialCells: 20
  verticalCells: 20
  spacing: nested
  sampling: average
  components:
    disk:
      stellar: {profile: exponentialDisk, verticalStructure: sechSquared,
                scaleRadial: 1.0, scaleHeight: 0.137}
      dust: {profile: exponentialDisk, verticalStructure: sechSquared,
             scaleRadial: 1.0, scaleHeight: 0.137,
             opticalDepth: {minimum: 1.0, maximum: 10.0, count: 2, includeZero: true}}
    spheroid:
      stellar: {profile: hernquist, truncation: 5.8,
                scaleRadial: {minimum: 0.1, maximum: 1.0, count: 2}}
tabulation:
  wavelengths: {minimum: 0.44, maximum: 0.55, count: 2}
  inclinations: {minimum: 0.0, maximum: 90.0, count: 2, spacing: linear}
  photons: 2000
"""


@pytest.fixture
def config(tmp_path):
    path = tmp_path / "tiny.yaml"
    path.write_text(TINY, encoding="utf-8")
    return path


@pytest.fixture
def runner():
    return CliRunner()


def test_version(runner):
    from dustcompendium import __version__

    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


class TestValidate:
    def test_it_describes_the_campaign(self, runner, config):
        result = runner.invoke(app, ["validate", str(config)])
        assert result.exit_code == 0, result.stdout
        assert "tiny" in result.stdout
        assert "opticalDepth:disk" in result.stdout
        assert "scaleRadial:spheroid" in result.stdout
        assert "9 models" in result.stdout

    def test_it_reports_the_grid_settings(self, runner, config):
        result = runner.invoke(app, ["validate", str(config)])
        assert "nested spacing" in result.stdout
        assert "average sampling" in result.stdout

    def test_it_needs_no_hyperion(self, runner, config, monkeypatch):
        """Validating is the cheap check, and must not need a radiative transfer install."""
        import builtins

        real_import = builtins.__import__

        def refuse(name, *args, **kwargs):
            if name == "hyperion" or name.startswith("hyperion."):
                raise ImportError("hyperion is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", refuse)
        result = runner.invoke(app, ["validate", str(config)])
        assert result.exit_code == 0, result.stdout

    def test_a_missing_file_is_reported(self, runner, tmp_path):
        result = runner.invoke(app, ["validate", str(tmp_path / "absent.yaml")])
        assert result.exit_code != 0

    def test_an_invalid_configuration_is_reported(self, runner, tmp_path):
        path = tmp_path / "bad.yaml"
        path.write_text("label: bad\n", encoding="utf-8")
        result = runner.invoke(app, ["validate", str(path)])
        assert result.exit_code != 0


@pytest.mark.hyperion
class TestBuild:
    def test_it_writes_one_input_per_model(self, runner, config, tmp_path):
        output = tmp_path / "models"
        result = runner.invoke(app, ["build", str(config), "-o", str(output)])
        assert result.exit_code == 0, result.stdout
        written = sorted(path.name for path in output.glob("*.hdf5"))
        assert len(written) == 9
        assert "tiny_disk_0.hdf5" in written
        assert "tiny_spheroid_2_1.hdf5" in written

    def test_it_resumes_rather_than_rewriting(self, runner, config, tmp_path):
        output = tmp_path / "models"
        runner.invoke(app, ["build", str(config), "-o", str(output), "--limit", "2"])
        result = runner.invoke(app, ["build", str(config), "-o", str(output)])
        assert result.exit_code == 0, result.stdout
        assert "skipped 2" in result.stdout
        assert len(list(output.glob("*.hdf5"))) == 9

    def test_the_limit_is_honoured(self, runner, config, tmp_path):
        output = tmp_path / "models"
        result = runner.invoke(app, ["build", str(config), "-o", str(output), "--limit", "3"])
        assert result.exit_code == 0, result.stdout
        assert len(list(output.glob("*.hdf5"))) == 3
