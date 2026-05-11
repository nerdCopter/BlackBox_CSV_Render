## Code Changes
- Preserve comments unless related code changes
- Add only functional comments, no AI/history notes
- Ask before removing any debug console output
- **All constants go in `src/constants.rs`** — no hardcoded magic numbers in function code
  - **Exception:** Axis indices (0=Roll, 1=Pitch, 2=Yaw) are managed via `src/axis_names.rs` module with `AXIS_COUNT`, `AXIS_NAMES`, and `axis_name()` function. Use these instead of creating redundant axis index constants.
  - See `src/axis_names.rs` for centralized axis naming (commit 73f8c04)
- Run checks in this order:
  1) `cargo clippy --all-targets --all-features -- -D warnings` — fix all warnings.
  2) `cargo fmt --all` — only after clippy passes.

## Firmware Detection
- **ALWAYS** use `Firmware revision` header to detect firmware type
- **NEVER** use `Firmware type` (unreliable - Emuflight reports as "Cleanflight")
- Firmware revision patterns:
  - Betaflight: `"Betaflight 4.x.x"` or `"Betaflight YYYY.MM"` or similar
  - Emuflight: `"EmuFlight 0.x.x"` or similar  
  - INAV: `"INAV x.x.x"` or similar

## Testing
- **ALWAYS** prefer `cargo build --release` rather than `cargo run`
- **ALWAYS** use `--output-dir` ./output/
- Temporary test code must go in ./tests/

## Git Workflow  
- **NEVER** use `git add -A` or `git add .`
- **NEVER** automatically commit changes without an explicit user request
- Only stage modified files and specified files when commanded
- Update `.gitignore` when required
- **ONLY** commit when explicitly commanded with phrases such as:
  - "commit"
  - "git commit"
  - "commit changes" 
  - "git add and commit"
  - "stage and commit"
- Generate concise commit messages from `git diff HEAD`

## Development Setup
- Use `.github/setup-dev.sh` for environment setup
- Pre-commit hook enforces formatting and clippy checks
- All changes must pass CI pipeline
## Documentation
- **OVERVIEW.md Maintenance:** For new features, add documentation in `OVERVIEW.md` at proper, appropriate locations
  - Information should be balanced or concise; never overly verbose
  - Align verbosity with existing documentation style in OVERVIEW.md
  - Verify placement fits the document structure and flow
- Update Table of Contents in OVERVIEW.md when adding new sections
- Keep README.md usage/examples synchronized with actual CLI flags in `src/main.rs`

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **BlackBox_CSV_Render** (817 symbols, 1894 relationships, 70 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/BlackBox_CSV_Render/context` | Codebase overview, check index freshness |
| `gitnexus://repo/BlackBox_CSV_Render/clusters` | All functional areas |
| `gitnexus://repo/BlackBox_CSV_Render/processes` | All execution flows |
| `gitnexus://repo/BlackBox_CSV_Render/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
