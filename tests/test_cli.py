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


class TestRun:
    def test_missing_inputs_are_reported_and_fail(self, runner, config, tmp_path):
        result = runner.invoke(
            app, ["run", str(config), "-m", str(tmp_path / "absent"), "-o", str(tmp_path / "out")]
        )
        assert result.exit_code == 1
        assert "model inputs are missing" in result.output
        assert "dust-compendium build" in result.output

    def test_an_already_solved_campaign_does_nothing(self, runner, config, tmp_path):
        """Resuming must not need the solver, or the inputs, for work already done."""
        import h5py

        from dustcompendium.campaign import Campaign
        from dustcompendium.config import load_campaign

        output = tmp_path / "out"
        output.mkdir()
        campaign = Campaign(load_campaign(str(config)))
        for candidate in campaign.runs():
            with h5py.File(output / f"{candidate.file_stem}.rtout", "w") as handle:
                handle.create_group("Peeled/group_00001")
        result = runner.invoke(app, ["run", str(config), "-m", str(tmp_path / "absent"), "-o", str(output)])
        assert result.exit_code == 0, result.output
        assert "all 9 models are already solved" in result.output

    def test_an_output_with_no_seds_is_solved_again(self, runner, config, tmp_path):
        """Hyperion aborts and still exits zero, leaving an empty peeled group.

        Counting such a file as solved would skip it on every later run, and the
        gap would surface only when the tabulation was assembled.
        """
        import h5py

        from dustcompendium.campaign import Campaign
        from dustcompendium.config import load_campaign

        output = tmp_path / "out"
        output.mkdir()
        campaign = Campaign(load_campaign(str(config)))
        for candidate in campaign.runs():
            with h5py.File(output / f"{candidate.file_stem}.rtout", "w") as handle:
                handle.create_group("Peeled")
        result = runner.invoke(app, ["run", str(config), "-m", str(tmp_path / "absent"), "-o", str(output)])
        assert result.exit_code == 1
        assert "model inputs are missing" in result.output

    @pytest.mark.hyperion
    @pytest.mark.solver
    @pytest.mark.slow
    def test_the_whole_pipeline_builds_then_runs(self, runner, config, tmp_path):
        models, output = tmp_path / "models", tmp_path / "out"
        built = runner.invoke(app, ["build", str(config), "-o", str(models)])
        assert built.exit_code == 0, built.output

        solved = runner.invoke(
            app,
            ["run", str(config), "-m", str(models), "-o", str(output), "--concurrency", "4"],
        )
        assert solved.exit_code == 0, solved.output
        assert "9 of 9 models succeeded" in solved.output
        assert len(list(output.glob("*.rtout"))) == 9
        assert len(list((output / "logs").glob("*.log"))) == 9

        again = runner.invoke(app, ["run", str(config), "-m", str(models), "-o", str(output)])
        assert "already solved" in again.output

    def test_a_task_count_which_does_not_divide_the_nodes_is_refused(self, runner, config, tmp_path):
        """Silently rounding down would hand the job fewer tasks than were asked for."""
        result = runner.invoke(
            app,
            [
                "run",
                str(config),
                "-m",
                str(tmp_path),
                "-o",
                str(tmp_path / "out"),
                "--tasks",
                "6",
                "--nodes",
                "4",
            ],
        )
        assert result.exit_code == 1
        assert "does not divide evenly" in result.output

    def test_matching_tasks_and_nodes_are_accepted(self, runner, config, tmp_path):
        result = runner.invoke(
            app,
            [
                "run",
                str(config),
                "-m",
                str(tmp_path / "absent"),
                "-o",
                str(tmp_path / "out"),
                "--tasks",
                "64",
                "--nodes",
                "4",
            ],
        )
        # Fails on the missing inputs, not on the task count.
        assert "does not divide evenly" not in result.output

    def test_an_unknown_scheduler_is_refused_before_any_work(self, runner, config, tmp_path):
        """Checked up front, so a typo does not traceback after building the job list."""
        result = runner.invoke(
            app,
            ["run", str(config), "-m", str(tmp_path), "-o", str(tmp_path / "out"), "-s", "slrum"],
        )
        assert result.exit_code == 1
        assert "unknown scheduler" in result.output
        assert "local, slurm" in result.output
