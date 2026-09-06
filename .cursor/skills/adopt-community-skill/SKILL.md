---
name: adopt-community-skill
description: Vet and install one community Agent Skill (e.g. from VoltAgent/awesome-agent-skills) into midnightodyssey/dev-standards for universal use. Use when adding, updating, or rejecting a third-party SKILL.md package.
---

# Adopt a community Agent Skill

Goal: add **one** curated skill to the universal `dev-standards` Cursor plugin — never bulk-install an awesome list.

Source catalogue: [VoltAgent/awesome-agent-skills](https://github.com/VoltAgent/awesome-agent-skills) (compatible `SKILL.md` packages). Skills are curated upstream, **not** audited.

## Workflow

1. **Name the skill** and the upstream URL (repo + path to the folder that contains `SKILL.md`).
2. **Security pass**
   - Read `SKILL.md` and any `scripts/`, `references/`, hooks.
   - Reject skills that exfiltrate secrets, run opaque remote payloads, or demand blanket tool access without need.
   - Prefer official team skills over anonymous one-off repos.
3. **Fit pass**
   - Must help more than one `midnightodyssey` repo, or be clearly reusable.
   - Must not duplicate an existing plugin skill (`specialist-routing`, `python-house-style`, …).
   - Description must state *what* and *when* in third person with matchable keywords.
4. **Install into the plugin**
   - Copy the skill folder to:
     `plugins/dev-standards/skills/<skill-name>/`
   - Keep relative paths; strip machine-absolute paths.
   - Run `scripts/install-cursor-skills.sh` so `~/.cursor/skills/` and `.cursor/skills/` stay in sync.
5. **Record**
   - Append a row to `cursor/CURATED_SKILLS.md` (name, upstream, date, why kept, risks accepted).
6. **Ship**
   - Commit on a feature branch; open a PR to `dev-standards`.
   - If Team Marketplace is connected with Auto Refresh, push updates the indexed plugin.
   - Otherwise re-run the install script on each machine / enable Sync Skills for Cloud Agents.

## Reject / defer

- Entire-catalogue installs
- Skills overlapping house style already covered here
- Skills that only apply to one throwaway experiment (put those in that repo's `.cursor/skills/` instead)
