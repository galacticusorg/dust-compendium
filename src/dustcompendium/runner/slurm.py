r"""Running jobs through Slurm.

Follows the shape of Galacticus's own ``python/queueManager.py``: write a batch
script per job, submit with ``sbatch``, and watch ``squeue`` and ``sacct`` until
they finish. It is reimplemented here rather than imported because that module
needs ``GALACTICUS_EXEC_PATH`` and a ``galacticusConfig.xml``, which a package
installed from PyPI cannot assume.

Two differences from that implementation are deliberate. Only *this campaign's*
jobs count towards the concurrency limit, rather than everything the user has
queued, so a busy account does not stall the campaign. And ``sacct`` is only
called when there is something to ask it about.
"""

import json
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from pathlib import Path

from .base import Job, JobResult, Scheduler

__all__ = ["SlurmScheduler", "batch_script"]

#: Slurm states meaning a job has not finished yet.
ACTIVE_STATES = frozenset({"RUNNING", "PENDING", "CONFIGURING", "COMPLETING", "REQUEUED"})


def batch_script(job: Job) -> str:
    """The ``sbatch`` script for a job."""
    resources = job.resources
    directives = [
        ("job-name", job.label),
        ("output", str(job.log_file)),
        ("error", str(job.log_file)),
        ("nodes", resources.nodes),
        ("ntasks-per-node", resources.tasks_per_node),
        ("cpus-per-task", resources.cpus_per_task),
        ("partition", resources.partition),
        ("time", resources.walltime),
        ("mem-per-cpu", None if resources.memory_per_cpu is None else f"{resources.memory_per_cpu}M"),
    ]
    lines = ["#!/bin/bash"]
    lines += [f"#SBATCH --{name}={value}" for name, value in directives if value is not None]
    lines += [
        "ulimit -t unlimited",
        "ulimit -c unlimited",
        " ".join(job.command),
    ]
    return "\n".join(lines) + "\n"


class SlurmScheduler(Scheduler):
    """Submit jobs to Slurm and wait for them.

    Parameters
    ----------
    concurrency
        How many of this campaign's jobs to keep queued or running at once.
    script_directory
        Where to write the batch scripts. Defaults to beside each job's log.
    poll_interval
        Seconds between checks on the queue.
    submit_interval
        Seconds to wait between submissions, to be kind to the scheduler.
    submit_attempts
        How many times to retry a failing ``sbatch`` before giving up, backing
        off between attempts.
    """

    def __init__(
        self,
        concurrency: int = 1,
        script_directory: Path | None = None,
        poll_interval: float = 10.0,
        submit_interval: float = 1.0,
        submit_attempts: int = 10,
    ) -> None:
        super().__init__(concurrency)
        self.script_directory = None if script_directory is None else Path(script_directory)
        self.poll_interval = poll_interval
        self.submit_interval = submit_interval
        self.submit_attempts = submit_attempts

    def _command(self, argv: Sequence[str]) -> subprocess.CompletedProcess:
        return subprocess.run(list(argv), capture_output=True, text=True, check=False)

    def _script_path(self, job: Job) -> Path:
        directory = self.script_directory or job.log_file.parent
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{job.label}.sbatch"

    def _submit(self, job: Job) -> str | None:
        """Submit a job, returning its Slurm id, or ``None`` if it would not go."""
        path = self._script_path(job)
        path.write_text(batch_script(job), encoding="utf-8")
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(self.submit_attempts):
            result = self._command(["sbatch", str(path)])
            if result.returncode == 0:
                found = re.search(r"\d+", result.stdout)
                if found is not None:
                    return found.group(0)
                return None
            if attempt + 1 < self.submit_attempts:
                time.sleep(max(self.submit_interval, 1.0) * 2.0**attempt)
        return None

    def _active(self, identifiers: set[str]) -> set[str]:
        """Which of the given jobs Slurm still considers unfinished."""
        result = self._command(["squeue", "--me", "--json"])
        if result.returncode != 0:
            # A transient failure must not be read as "everything finished".
            return set(identifiers)
        payload = json.loads(result.stdout)
        active = set()
        for entry in payload.get("jobs", []):
            identifier = str(entry["job_id"])
            states = entry.get("job_state") or []
            state = states[0] if isinstance(states, list) else states
            if identifier in identifiers and state in ACTIVE_STATES:
                active.add(identifier)
        return active

    def _statuses(self, identifiers: set[str]) -> dict[str, int]:
        """Exit status of each finished job, by Slurm id."""
        if not identifiers:
            return {}
        result = self._command(["sacct", "-j", ",".join(sorted(identifiers)), "--json"])
        if result.returncode != 0:
            return dict.fromkeys(identifiers, -1)
        payload = json.loads(result.stdout)
        statuses: dict[str, int] = {}
        for entry in payload.get("jobs", []):
            identifier = str(entry["job_id"])
            if identifier not in identifiers:
                continue
            try:
                statuses[identifier] = int(entry["exit_code"]["return_code"]["number"])
            except (KeyError, TypeError, ValueError):
                statuses[identifier] = -2
        # Anything sacct said nothing about finished without leaving a record.
        return {identifier: statuses.get(identifier, -1) for identifier in identifiers}

    def run(
        self,
        jobs: Sequence[Job],
        on_complete: Callable[[JobResult], None] | None = None,
    ) -> list[JobResult]:
        if not jobs:
            return []
        pending = list(jobs)
        submitted: dict[str, Job] = {}
        results: dict[str, JobResult] = {}

        def finish(result: JobResult) -> None:
            results[result.job.label] = result
            if on_complete is not None:
                on_complete(result)

        while pending or submitted:
            while pending and len(submitted) < self.concurrency:
                job = pending.pop(0)
                identifier = self._submit(job)
                if identifier is None:
                    finish(JobResult(job, -3))
                    continue
                submitted[identifier] = job
                if pending:
                    time.sleep(self.submit_interval)
            if not submitted:
                continue
            active = self._active(set(submitted))
            done = set(submitted) - active
            if not done:
                time.sleep(self.poll_interval)
                continue
            for identifier, status in self._statuses(done).items():
                finish(JobResult(submitted.pop(identifier), status))
        return [results[job.label] for job in jobs]
