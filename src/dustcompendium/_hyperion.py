"""Reporting a missing Hyperion in a way that says what to do about it.

Hyperion is not a dependency of this package -- no released version works on
NumPy 2 -- so it has to be installed separately, and the commonest way for that
to go wrong is having it in one environment and running from another. The
message therefore names the interpreter it looked in, which is the fact that
identifies the mistake.
"""

import sys

__all__ = ["missing_hyperion"]

#: Which commands need it, so the message can say whether the one being run does.
NEEDED_BY = "`build` and `collect` need it; `validate`, `run`, `plot` and `compare` do not"


def missing_hyperion(action: str) -> ImportError:
    """The error to raise when Hyperion is needed and absent."""
    return ImportError(
        f"{action} needs Hyperion, which is not installed in {sys.executable}. "
        f"{NEEDED_BY}. If it is installed elsewhere, you are running from the "
        "wrong environment. Otherwise see the installation notes: no released "
        "version can build models on NumPy 2, so it has to come from source."
    )
