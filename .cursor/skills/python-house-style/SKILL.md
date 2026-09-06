---
name: python-house-style
description: Apply the Python house style when writing or reviewing Python. Use for any Python edit, refactor, review, or new module.
---

# Python house style

Canonical prose lives in this `dev-standards` repo:

- `STYLE.md` — full guide (section 0 is the short version)
- `DELTAS.md` — where existing code still disagrees; do not treat non-compliance as precedent
- `python/ruff.toml` and `python/pyproject.template.toml` — tooling defaults

Locate them in this order: current workspace copies → sibling `../dev-standards/` → clone of `midnightodyssey/dev-standards`.

## Non-negotiables

1. **`Decimal` for every money and rate value. Never `float`.** Enforce at model boundaries.
2. **Engine never imports UI.** `<core>` is pure; `<core>_ui` depends on it, never reverse.
3. **Signature-annotated everywhere; `mypy --strict` clean** over both packages when the repo is set up for it.
4. **`log.debug("... %s", value)`** — `%`-style in logging calls; f-strings elsewhere.
5. **No bare `except`.** Wrap and re-raise as a typed app error with `from exc`.
6. **Comments explain why, never what.**
7. **Refuse to guess.** Surface ambiguity instead of picking the first match.

## Mechanics

- Line length / format: shared ruff config (100), not Black 88.
- Prefer `X | None` over `Optional[X]`.
- Absolute first-party imports (ruff `TID252`).
- Google docstrings for new code.
- Trailing commas on multi-line literals (ruff format).

## When reviewing

Call out house-style violations that matter for correctness first (Decimal, exception handling, engine/UI boundary). Style-only nits second.
