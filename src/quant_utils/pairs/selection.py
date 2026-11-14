"""Pair selection with a Bonferroni correction over every candidate."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_utils.pairs.cointegration import DEFAULT_ADF_LAGS, engle_granger
from quant_utils.pairs.errors import PairsError
from quant_utils.pairs.spread import half_life

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PairCandidate:
    """A pair that passed selection. ``symbol_a`` is the y leg."""

    symbol_a: str
    symbol_b: str
    tau: float
    pvalue: float
    alpha: float
    beta: float
    half_life: float
    n_obs: int


@dataclass(frozen=True, slots=True)
class PairSelection:
    """Result of :func:`select_pairs`.

    ``n_candidates`` counts every pair submitted, including ones that could not
    be tested. It is the Bonferroni divisor and the trial count for any
    downstream multiple-testing accounting: a pair that was merely looked at is
    a trial.
    """

    n_candidates: int
    n_tested: int
    alpha: float
    alpha_adjusted: float
    selected: tuple[PairCandidate, ...]


def candidate_pairs(groups: Mapping[str, Sequence[str]]) -> list[tuple[str, str]]:
    """All within-group symbol pairs, each sorted ``(a, b)`` with ``a < b``.

    A symbol in two groups is ambiguous and raises instead of being silently
    assigned to the first.
    """
    seen: dict[str, str] = {}
    for name, members in groups.items():
        for sym in members:
            if sym in seen:
                raise PairsError(f"{sym} is in both group {seen[sym]!r} and {name!r}")
            seen[sym] = name
    out: list[tuple[str, str]] = []
    for members in groups.values():
        out.extend(itertools.combinations(sorted(members), 2))
    return out


def select_pairs(
    prices: pd.DataFrame,
    candidates: Sequence[tuple[str, str]],
    *,
    half_life_min: float,
    half_life_max: float,
    beta_min: float,
    beta_max: float,
    alpha: float = 0.05,
    lags: int = DEFAULT_ADF_LAGS,
) -> PairSelection:
    """Engle–Granger selection with Bonferroni over ALL candidates.

    ``prices`` are price LEVELS (logged here) from a formation window only; do
    not pass bars you will later trade. A pair is selected iff
    ``p <= alpha / n_candidates``, its half-life is within
    ``[half_life_min, half_life_max]`` and its hedge ratio within
    ``[beta_min, beta_max]``. The bounds are required: the library has no
    opinion on them. Each pair is tested once, in the order given (``a``
    regressed on ``b``), never in both orders.

    The p-value is the approximation in :func:`eg_pvalue`; it is conservative
    below 1%, which is where a Bonferroni threshold often lands.
    """
    k = len(candidates)
    adj = alpha / k if k else alpha
    selected: list[PairCandidate] = []
    tested = 0
    for a, b in candidates:
        if a not in prices.columns or b not in prices.columns:
            logger.warning("pair (%s, %s) not testable: missing price column", a, b)
            continue
        la, lb = np.log(prices[a]), np.log(prices[b])
        try:
            res = engle_granger(la, lb, use_log=False, lags=lags)
            hl = half_life(res.residuals)
        except (PairsError, ValueError) as exc:
            logger.warning("pair (%s, %s) not testable: %s", a, b, exc)
            continue
        tested += 1
        if res.adf_pvalue_approx > adj or not (half_life_min <= hl <= half_life_max):
            continue
        if not (beta_min <= res.hedge_ratio <= beta_max):
            continue
        selected.append(
            PairCandidate(
                a,
                b,
                res.adf_stat,
                res.adf_pvalue_approx,
                res.intercept,
                res.hedge_ratio,
                hl,
                res.nobs,
            )
        )
    selected.sort(key=lambda c: c.pvalue)
    return PairSelection(
        n_candidates=k,
        n_tested=tested,
        alpha=alpha,
        alpha_adjusted=adj,
        selected=tuple(selected),
    )
