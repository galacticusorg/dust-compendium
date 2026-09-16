r"""Running jobs on this machine.

A pool of concurrent subprocesses. Each job's output goes to its own log file,
so a failure can be read afterwards rather than being interleaved with every
other job's.

Threads rather than processes: each worker only starts a subprocess and waits
for it, which releases the interpreter lock, and the work itself is in the
solver.
"""

import subprocess
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from .base import Job, JobResult, Scheduler

__all__ = ["LocalScheduler"]


class LocalScheduler(Scheduler):
    """Run jobs as subprocesses on this machine.

    Parameters
    ----------
    concurrency
        How many to run at once. Each solver is itself serial, so this is how
        many cores the campaign will occupy.
    """

    def _run_one(self, job: Job) -> JobResult:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        with job.log_file.open("w", encoding="utf-8") as log:
            log.write(f"# {' '.join(job.command)}\n")
            log.flush()
            try:
                completed = subprocess.run(
                    list(job.command), stdout=log, stderr=subprocess.STDOUT, check=False
                )
            except OSError as error:
                log.write(f"\nfailed to start: {error}\n")
                return JobResult(job, -3)
        return JobResult(job, completed.returncode)

    def run(
        self,
        jobs: Sequence[Job],
        on_complete: Callable[[JobResult], None] | None = None,
    ) -> list[JobResult]:
        if not jobs:
            return []
        results: dict[str, JobResult] = {}
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            futures = {pool.submit(self._run_one, job): job for job in jobs}
            for future in as_completed(futures):
                result = future.result()
                results[result.job.label] = result
                if on_complete is not None:
                    on_complete(result)
        return [results[job.label] for job in jobs]
