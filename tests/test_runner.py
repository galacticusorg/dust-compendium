"""Tests for the schedulers and the solver command."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from dustcompendium.runner import (
    SCHEDULERS,
    Job,
    JobResult,
    LocalScheduler,
    Resources,
    Scheduler,
    SlurmScheduler,
    scheduler,
    solver_command,
    solver_copies,
    which_launcher,
    which_solver,
)
from dustcompendium.runner.slurm import batch_script


def echo_job(tmp_path, label="job", message="hello", status=0):
    """A job which needs no solver: this interpreter, printing and exiting."""
    script = f"import sys; print({message!r}); sys.exit({status})"
    return Job(label, [sys.executable, "-c", script], tmp_path / f"{label}.log")


class TestJob:
    def test_a_job_needs_a_label_and_a_command(self, tmp_path):
        with pytest.raises(ValueError, match="needs a label"):
            Job("", ["true"], tmp_path / "log")
        with pytest.raises(ValueError, match="has no command"):
            Job("j", [], tmp_path / "log")

    def test_resources_reject_nonsense(self):
        with pytest.raises(ValueError, match="nodes must be at least one"):
            Resources(nodes=0)

    def test_tasks_is_nodes_times_tasks_per_node(self):
        assert Resources(nodes=4, tasks_per_node=16).tasks == 64

    def test_failure_messages_name_the_log(self, tmp_path):
        job = echo_job(tmp_path)
        assert str(job.log_file) in JobResult(job, 1).failure_message()
        assert "never submitted" in JobResult(job, -3).failure_message()


class TestSolverCommand:
    def test_the_serial_command_takes_no_launcher(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        command = solver_command(Path("in.hdf5"), Path("out.rtout"))
        assert command == ["/bin/hyperion_cyl", "-f", "in.hdf5", "out.rtout"]

    def test_the_parallel_command_uses_the_launcher_and_mpi_binary(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        command = solver_command(Path("in.hdf5"), Path("out.rtout"), tasks=64, launcher=("mpirun", "-np"))
        assert command[:4] == ["mpirun", "-np", "64", "/bin/hyperion_cyl_mpi"]

    def test_the_default_launcher_is_the_one_matching_the_binary(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        monkeypatch.setattr(
            "dustcompendium.runner.solver.which_launcher", lambda binary: [f"{binary}.launcher", "-n"]
        )
        command = solver_command(Path("in.hdf5"), Path("out.rtout"), tasks=8)
        assert command[:4] == ["/bin/hyperion_cyl_mpi.launcher", "-n", "8", "/bin/hyperion_cyl_mpi"]

    def test_overwrite_can_be_turned_off(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        assert "-f" not in solver_command(Path("a"), Path("b"), overwrite=False)

    def test_a_missing_binary_points_at_the_build_notes(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="pip does not install"):
            which_solver()

    def test_an_unknown_geometry_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="cartesian"):
            which_solver("toroidal")

    def test_zero_tasks_is_rejected(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda name: f"/bin/{name}")
        with pytest.raises(ValueError, match="at least one"):
            solver_command(Path("a"), Path("b"), tasks=0)


def fake_ldd(monkeypatch, libmpi):
    """Make ``ldd`` report that a binary loads the given MPI library."""
    listing = (
        f"\tlibmpi.so.40 => {libmpi} (0x00007f6e0a533000)\n"
        if libmpi
        else "\tlibc.so.6 => /lib64/libc.so.6\n"
    )

    def run(command, **options):
        assert command[0] == "ldd"
        return subprocess.CompletedProcess(command, 0, stdout=listing, stderr="")

    monkeypatch.setattr("dustcompendium.runner.solver.subprocess.run", run)


class TestWhichLauncher:
    """The launcher has to come from the MPI installation the solver was linked against.

    Any other one starts each task as its own single-rank job, so that sixteen
    copies of the solver each run the whole model and race to write one file.
    """

    def test_it_is_found_beside_the_mpi_library(self, monkeypatch, tmp_path):
        (tmp_path / "lib").mkdir()
        (tmp_path / "bin").mkdir()
        mpirun = tmp_path / "bin" / "mpirun"
        mpirun.write_text("#!/bin/sh\n")
        mpirun.chmod(0o755)
        fake_ldd(monkeypatch, tmp_path / "lib" / "libmpi.so.40")
        monkeypatch.setattr("shutil.which", lambda name: "/elsewhere/bin/mpirun")
        assert which_launcher("solver") == [str(mpirun), "-np"]

    def test_it_falls_back_to_path_as_an_absolute_path(self, monkeypatch, tmp_path):
        fake_ldd(monkeypatch, None)
        monkeypatch.setattr("shutil.which", lambda name: "relative/bin/mpirun")
        launcher = which_launcher("solver")
        assert Path(launcher[0]).is_absolute()
        assert launcher[0].endswith("relative/bin/mpirun")

    def test_a_library_with_no_launcher_beside_it_falls_back_to_path(self, monkeypatch, tmp_path):
        fake_ldd(monkeypatch, tmp_path / "lib" / "libmpi.so.40")
        monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/mpirun")
        assert which_launcher("solver") == ["/usr/bin/mpirun", "-np"]

    def test_no_launcher_at_all_is_an_error(self, monkeypatch):
        fake_ldd(monkeypatch, None)
        monkeypatch.setattr("shutil.which", lambda name: None)
        with pytest.raises(FileNotFoundError, match="--launcher"):
            which_launcher("solver")


class TestSolverCopies:
    BANNER = " Hyperion v1.0.0\n Started on 13 October 2026 at 15:31:02\n Input: in.hdf5\n"

    def test_a_correct_mpi_run_has_one_banner(self, tmp_path):
        log = tmp_path / "log"
        log.write_text(self.BANNER + " WARNING: no dust present\n" * 2)
        assert solver_copies(log) == 1

    def test_each_independent_copy_has_its_own_banner(self, tmp_path):
        log = tmp_path / "log"
        log.write_text(self.BANNER * 16)
        assert solver_copies(log) == 16

    def test_a_missing_log_counts_as_none(self, tmp_path):
        assert solver_copies(tmp_path / "absent") == 0


class TestLocalScheduler:
    def test_it_runs_a_job_and_reports_success(self, tmp_path):
        results = LocalScheduler().run([echo_job(tmp_path)])
        assert len(results) == 1 and results[0].succeeded

    def test_it_captures_output_to_the_log(self, tmp_path):
        job = echo_job(tmp_path, message="a distinctive line")
        LocalScheduler().run([job])
        assert "a distinctive line" in job.log_file.read_text(encoding="utf-8")
        assert job.log_file.read_text(encoding="utf-8").startswith("# ")

    def test_it_reports_a_failing_job(self, tmp_path):
        results = LocalScheduler().run([echo_job(tmp_path, status=3)])
        assert not results[0].succeeded
        assert results[0].status == 3

    def test_a_command_which_cannot_start_is_reported(self, tmp_path):
        job = Job("absent", ["/no/such/binary"], tmp_path / "absent.log")
        results = LocalScheduler().run([job])
        assert results[0].status == -3
        assert "failed to start" in job.log_file.read_text(encoding="utf-8")

    def test_results_come_back_in_the_order_given(self, tmp_path):
        jobs = [echo_job(tmp_path, label=f"job{index}") for index in range(6)]
        results = LocalScheduler(concurrency=4).run(jobs)
        assert [result.job.label for result in results] == [job.label for job in jobs]

    def test_every_job_runs_when_concurrency_exceeds_one(self, tmp_path):
        jobs = [echo_job(tmp_path, label=f"job{index}") for index in range(8)]
        results = LocalScheduler(concurrency=4).run(jobs)
        assert all(result.succeeded for result in results)
        assert all(job.log_file.exists() for job in jobs)

    def test_the_completion_hook_sees_every_job(self, tmp_path):
        seen = []
        jobs = [echo_job(tmp_path, label=f"job{index}") for index in range(5)]
        LocalScheduler(concurrency=2).run(jobs, on_complete=seen.append)
        assert {result.job.label for result in seen} == {job.label for job in jobs}

    def test_an_empty_list_is_fine(self):
        assert LocalScheduler().run([]) == []

    def test_concurrency_must_be_positive(self):
        with pytest.raises(ValueError, match="at least one"):
            LocalScheduler(concurrency=0)


class TestBatchScript:
    def test_it_carries_the_resources_it_was_given(self, tmp_path):
        job = Job(
            "model",
            ["mpirun", "-np", "64", "hyperion_cyl_mpi", "in", "out"],
            tmp_path / "model.log",
            Resources(
                nodes=4,
                tasks_per_node=16,
                partition="compute",
                walltime="02:00:00",
                memory_per_cpu=2048,
            ),
        )
        script = batch_script(job)
        assert "#SBATCH --nodes=4" in script
        assert "#SBATCH --ntasks-per-node=16" in script
        assert "#SBATCH --partition=compute" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --mem-per-cpu=2048M" in script
        assert script.splitlines()[-1] == "mpirun -np 64 hyperion_cyl_mpi in out"

    def test_unset_resources_are_left_out(self, tmp_path):
        script = batch_script(echo_job(tmp_path))
        assert "--partition" not in script
        assert "--time" not in script
        assert "--mem-per-cpu" not in script

    def test_it_starts_with_a_shebang(self, tmp_path):
        assert batch_script(echo_job(tmp_path)).startswith("#!/bin/bash\n")


class FakeSlurm:
    """A stand-in for sbatch, squeue and sacct.

    Jobs go straight from submitted to finished, with the exit status given, so
    a scheduler can be exercised without a cluster.
    """

    def __init__(self, statuses=None, submit_fails=False):
        self.statuses = statuses or {}
        self.submit_fails = submit_fails
        self.submitted: list[str] = []
        self.scripts: list[Path] = []
        self._next = 1000

    def __call__(self, argv, **kwargs):
        program = argv[0]
        if program == "sbatch":
            if self.submit_fails:
                return subprocess.CompletedProcess(argv, 1, "", "refused")
            self.scripts.append(Path(argv[1]))
            self._next += 1
            self.submitted.append(str(self._next))
            return subprocess.CompletedProcess(argv, 0, f"Submitted batch job {self._next}", "")
        if program == "squeue":
            return subprocess.CompletedProcess(argv, 0, json.dumps({"jobs": []}), "")
        if program == "sacct":
            identifiers = argv[2].split(",")
            jobs = [
                {
                    "job_id": int(identifier),
                    "exit_code": {"return_code": {"number": self.statuses.get(identifier, 0)}},
                }
                for identifier in identifiers
            ]
            return subprocess.CompletedProcess(argv, 0, json.dumps({"jobs": jobs}), "")
        raise AssertionError(f"unexpected command {argv}")


class TestSlurmScheduler:
    def make(self, tmp_path, fake, **options):
        one = SlurmScheduler(poll_interval=0.0, submit_interval=0.0, **options)
        one._command = fake
        return one

    def test_it_submits_and_collects_every_job(self, tmp_path):
        fake = FakeSlurm()
        jobs = [echo_job(tmp_path, label=f"job{index}") for index in range(4)]
        results = self.make(tmp_path, fake, concurrency=2).run(jobs)
        assert len(fake.submitted) == 4
        assert all(result.succeeded for result in results)
        assert [result.job.label for result in results] == [job.label for job in jobs]

    def test_it_writes_a_script_for_each_job(self, tmp_path):
        fake = FakeSlurm()
        self.make(tmp_path, fake).run([echo_job(tmp_path)])
        assert fake.scripts and fake.scripts[0].exists()
        assert fake.scripts[0].read_text(encoding="utf-8").startswith("#!/bin/bash")

    def test_it_honours_the_concurrency_limit(self, tmp_path):
        """Only this campaign's jobs are counted, not everything the user has queued."""
        in_flight = []
        fake = FakeSlurm()
        original = fake.__call__

        def watching(argv, **kwargs):
            if argv[0] == "squeue":
                in_flight.append(len(fake.submitted) - len(watching.finished))
                watching.finished = list(fake.submitted)
            return original(argv, **kwargs)

        watching.finished = []
        jobs = [echo_job(tmp_path, label=f"job{index}") for index in range(6)]
        self.make(tmp_path, watching, concurrency=2).run(jobs)
        assert max(in_flight) <= 2

    def test_a_failing_job_is_reported_with_its_status(self, tmp_path):
        fake = FakeSlurm(statuses={"1001": 7})
        results = self.make(tmp_path, fake).run([echo_job(tmp_path)])
        assert results[0].status == 7

    def test_a_job_which_will_not_submit_is_reported(self, tmp_path):
        fake = FakeSlurm(submit_fails=True)
        results = self.make(tmp_path, fake, submit_attempts=2).run([echo_job(tmp_path)])
        assert results[0].status == -3
        assert "never submitted" in results[0].failure_message()

    def test_a_transient_squeue_failure_does_not_finish_jobs(self, tmp_path):
        """Otherwise a blip would be read as every job having completed."""
        one = SlurmScheduler(poll_interval=0.0, submit_interval=0.0)
        one._command = lambda argv, **kwargs: subprocess.CompletedProcess(argv, 1, "", "down")
        assert one._active({"1", "2"}) == {"1", "2"}

    def test_a_job_sacct_says_nothing_about_is_not_called_a_success(self, tmp_path):
        one = SlurmScheduler()
        one._command = lambda argv, **kwargs: subprocess.CompletedProcess(
            argv, 0, json.dumps({"jobs": []}), ""
        )
        assert one._statuses({"5"}) == {"5": -1}

    def test_an_empty_list_is_fine(self):
        assert SlurmScheduler().run([]) == []


class TestFactory:
    @pytest.mark.parametrize("name", ["local", "slurm"])
    def test_schedulers_resolve_by_name(self, name):
        assert isinstance(scheduler(name, concurrency=2), Scheduler)

    def test_an_unknown_scheduler_lists_the_known_ones(self):
        with pytest.raises(KeyError, match="local, slurm"):
            scheduler("pbs")

    def test_the_registry_matches_the_factory(self):
        assert set(SCHEDULERS) == {"local", "slurm"}


class TestRelinkInput:
    """Making a solved model's link to its input relative.

    The solver records whatever path it was handed, which is absolute, so
    without this the results can only be read from the directory they were
    solved in -- and the chain on to the dust file is only usable if the whole
    tree can be moved together.
    """

    def _output_linking_to(self, directory, target):
        import h5py

        source = directory / "models" / "model.hdf5"
        source.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(source, "w") as handle:
            handle["Grid"] = [1.0]
        result = directory / "out" / "model.rtout"
        result.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(result, "w") as handle:
            handle["Input"] = h5py.ExternalLink(str(target), "/")
        return source, result

    def test_an_absolute_link_becomes_relative(self, tmp_path):
        import h5py

        from dustcompendium.runner import relink_input

        source, result = self._output_linking_to(tmp_path, tmp_path / "models" / "model.hdf5")
        assert relink_input(result, source) is True
        with h5py.File(result, "r") as handle:
            assert handle.get("Input", getlink=True).filename == "../models/model.hdf5"
            # And it still resolves.
            assert "Grid" in handle["Input"]

    def test_the_results_survive_the_tree_being_moved(self, tmp_path):
        import shutil

        import h5py

        from dustcompendium.runner import relink_input

        source, result = self._output_linking_to(
            tmp_path / "here", tmp_path / "here" / "models" / "model.hdf5"
        )
        relink_input(result, source)
        shutil.move(tmp_path / "here", tmp_path / "there")
        with h5py.File(tmp_path / "there" / "out" / "model.rtout", "r") as handle:
            assert "Grid" in handle["Input"]

    def test_a_copied_input_has_no_link_to_rewrite(self, tmp_path):
        import h5py

        from dustcompendium.runner import relink_input

        result = tmp_path / "model.rtout"
        with h5py.File(result, "w") as handle:
            handle.create_group("Input")
        assert relink_input(result, tmp_path / "model.hdf5") is False

    def test_an_unreadable_output_is_not_an_error(self, tmp_path):
        from dustcompendium.runner import relink_input

        missing = tmp_path / "nothing.rtout"
        assert relink_input(missing, tmp_path / "model.hdf5") is False
