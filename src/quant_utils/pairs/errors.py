"""Exception type shared by the pairs modules."""

from __future__ import annotations


class PairsError(ValueError):
    """Invalid input to a pairs primitive (bad parameters, NaNs, too few bars).

    Subclasses :class:`ValueError`, so callers that caught ``ValueError`` from
    earlier releases keep working.
    """
