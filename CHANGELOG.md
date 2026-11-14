# Changelog

## 0.2.0

### Breaking behaviour changes (pairs)

- **The Engle-Granger verdict is stricter.** v0.1.0 ran a plain constant-ADF on
  the residual and judged it against plain-ADF critical values (5% at -2.86).
  An OLS residual needs a no-intercept ADF and the cointegration critical values
  (5% at about -3.34, MacKinnon 2010, N=2). v0.1.0 rejected the no-cointegration
  null on roughly one in four independent random-walk pairs at the 5% level;
  0.2.0 is close to nominal. Pairs that "passed" under 0.1.0 may now fail.
- **`PairsStrategy.run()` no longer looks ahead.** It used to fit the hedge
  ratio on the full sample and then trade that same sample. It now requires
  `formation_window`, fits on those bars only, and holds flat until after them.
  A position at bar `t` is a close-of-bar decision earning PnL from `t + 1`.
- **A NaN z-score now means flat** (`generate_signals`); it used to hold the
  previous position.
- `engle_granger`: ADF lags are fixed (`lags=1`) by default instead of the
  Schwert rule. `max_lag` no longer sets a fixed lag: use `lags=`, or
  `autolag="aic"|"bic"` with an optional `max_lag` cap. The critical values in
  the result are finite-sample and keyed 0.01/0.05/0.10.
- `residual_half_life` uses the exact form `-ln 2 / ln(1 + b)` (was
  `-ln 2 / b`).

### Added

- `PairsError` (a `ValueError`).
- `eg_critical_value`, `eg_pvalue` (documented approximation: close between 1%
  and 10%, conservative below 1%).
- `rolling_hedge_ratio` (trailing), `zscore` (trailing), `half_life`.
- `positions_from_z`: entry/exit/stop/time-stop state machine with lock-out;
  `entry_z`, `exit_z`, `stop_z` and `max_hold` are required, with no defaults.
- `candidate_pairs`, `select_pairs` (Bonferroni over every candidate, reports
  the candidate count; half-life and beta bounds are required).
- `PairsStrategy.generate(..., trade_from=)`.

### Development

- `statsmodels` added to the `dev` extra only, to cross-check tau and the
  critical-value table in the tests. It is not a runtime dependency.

## 0.1.0

Initial release.
