r"""What a scheduler is, and the jobs it runs.

A campaign is thousands of independent radiative transfer solves, so the only
thing a scheduler has to do is run a list of commands and say which of them
worked. Backends differ in *where* they run -- this machine, or a cluster -- not
in what they run.

Results are returned rather than accumulated through a callback. The original
passed a closure to its launcher which reached back into shared arrays as each
job finished; that coupled running to post-processing, and left no artifact on
disk if the driver died partway through thousands of models. Here each job
leaves its output file, and collecting them is a separate step which can be run
again at any time.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Job", "JobResult", "Resources", "Scheduler"]


@dataclass(frozen=True)
class Resources:
    """What a job needs. Backends which cannot honour a field ignore it."""

    nodes: int = 1
    tasks_per_node: int = 1
    cpus_per_task: int = 1
    memory_per_cpu: int | None = None
    """Megabytes per cpu."""
    walltime: str | None = None
    """As ``HH:MM:SS``, or any form the scheduler accepts."""
    partition: str | None = None

    def __post_init__(self) -> None:
        for name in ("nodes", "tasks_per_node", "cpus_per_task"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be at least one, got {getattr(self, name)}")

    @property
    def tasks(self) -> int:
        """Total tasks, which is what an MPI launcher wants."""
        return self.nodes * self.tasks_per_node


@dataclass(frozen=True)
class Job:
    """One command to run, and where to put what it says."""

    label: str
    command: Sequence[str]
    log_file: Path
    resources: Resources = field(default_factory=Resources)

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("a job needs a label")
        if not self.command:
            raise ValueError(f"job {self.label!r} has no command")
        object.__setattr__(self, "command", tuple(self.command))
        object.__setattr__(self, "log_file", Path(self.log_file))


@dataclass(frozen=True)
class JobResult:
    """How a job turned out."""

    job: Job
    status: int
    """Exit status; zero is success. Negative values are the scheduler's own,
    for a job which never ran or whose status could not be recovered."""

    @property
    def succeeded(self) -> bool:
        return self.status == 0

    def failure_message(self) -> str:
        """A one-line account of the failure, with a pointer to the log."""
        reason = {
            -1: "no exit status could be recovered",
            -2: "the scheduler reported it as failed",
            -3: "it was never submitted",
        }.get(self.status, f"exit status {self.status}")
        return f"{self.job.label}: {reason}; see {self.job.log_file}"


class Scheduler(ABC):
    """Runs jobs somewhere, and reports what happened.

    Parameters
    ----------
    concurrency
        How many jobs to have in flight at once.
    """

    def __init__(self, concurrency: int = 1) -> None:
        if concurrency < 1:
            raise ValueError(f"concurrency must be at least one, got {concurrency}")
        self.concurrency = concurrency

    @abstractmethod
    def run(
        self,
        jobs: Sequence[Job],
        on_complete: Callable[[JobResult], None] | None = None,
    ) -> list[JobResult]:
        """Run every job and return a result for each, in the order given.

        Parameters
        ----------
        jobs
            The jobs to run.
        on_complete
            Called with each result as it arrives, for progress reporting. It
            must not be relied on for correctness: the returned list is the
            record.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}(concurrency={self.concurrency})"
