# Agents — start here

All agent instructions and project doc pointers live under **`docs/`**.

1. **`docs/agents.md`** — full operating mode, research stance, stack hints.
2. **`docs/progress.md`** — **read before coding**; log of tries, facts, current focus.
3. **`docs/design-decisions.md`** — **check before coding**; ADRs for non-obvious choices.

## Editor-specific agent config

| Editor | Config location | Notes |
|--------|----------------|-------|
| **GitHub Copilot** (VS Code) | `.github/copilot-instructions.md` | Unified workspace instructions — all rules + skill in one file |
| **Cursor** | `.cursor/rules/*.mdc`, `.cursor/skills/routing-research/SKILL.md` | Split across rule files and skill file |

Both contain the same content — keep them in sync when updating agent behavior.
