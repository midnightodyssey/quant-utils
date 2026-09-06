---
name: specialist-routing
description: Pick the right Cursor Task subagent type (and optional curated community skill) for the work. Use at the start of substantive tasks, when fan-out planning, or when unsure which specialist to launch.
---

# Specialist routing (Cursor)

Route work to the best **available** Cursor Task type. Do not invent agent names that are not in this session's available subagent list.

Claude Code machines that also load `claude/CLAUDE.global.md` keep the VoltAgent catalogue there (`python-pro`, `code-reviewer`, …). This skill is the **Cursor** map. Same intent, different roster.

## Step 0 — detect what this session has

1. Read the available `subagent_type` / Task roster for this session. That list is the only authority.
2. If a preferred specialist below is missing, say so once, then use the closest built-in (`explore`, `generalPurpose`, or whatever review/debug types this session exposes).
3. Never have the parent do the specialist's work just because the ideal name is absent.

## Cursor Task map

| Work | Prefer | Fallback |
|---|---|---|
| Find files, symbols, "where is X" | `explore` | `generalPurpose` |
| Multi-step research / open-ended | `generalPurpose` | — |
| Broad repo architecture questions | `explore` then plan in parent | `generalPurpose` |
| Manual GUI / browser verification | `computerUse` (when available) | — |
| Review a screen recording | `videoReview` (when available) | — |
| Cursor product "how do I…" | `cursor-guide` (when available) | — |
| Failing PR CI check diagnosis | `ci-investigator` (when available) | `generalPurpose` |
| Isolated experiment / best-of-N | `best-of-n-runner` (when available) | — |

## Domain hints (still use Task types above)

| Domain | Guidance |
|---|---|
| Python / PySide / money-decimal code | Parent applies `python-house-style`; code edits via `generalPurpose` (or a stronger coding model if the user named one) |
| Security / credentials | Prefer security-minded prompts; never print secrets |
| Multi-repo portfolio / digests | Prefer an installed `chief-of-staff` skill when present; otherwise `generalPurpose` with portfolio context |
| Adopting a community Agent Skill | Use `adopt-community-skill` |

## Parallelism

- Independent strands → multiple Task calls in **one** parent turn.
- Default batch size: **three**.

## Community Agent Skills

Only load skills that are already installed (plugin, `~/.cursor/skills/`, or project `.cursor/skills/`).

- Do **not** pull the entire [awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) catalogue into context.
- To add one, follow `adopt-community-skill` and record it in `cursor/CURATED_SKILLS.md` in the `dev-standards` repo.
