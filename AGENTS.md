## Code Changes
- Preserve comments unless related code changes
- Add only functional comments, no AI/history notes
- Ask before removing any debug console output
- Run checks in this order:
  1) `cargo clippy --all-targets --all-features -- -D warnings` — fix all warnings.
  2) `cargo fmt --all` — only after clippy passes.

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
