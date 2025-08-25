## Code Changes
- Preserve comments unless related code changes
- Add only functional comments, no AI/history notes
- **ALWAYS** run `cargo fmt --all` after ANY code changes
- **ALWAYS** run `cargo clippy --all-targets --all-features -- -D warnings` before changes
- **NEVER** skip formatting or clippy checks

## Git Workflow  
- **NEVER** use `git add -A` or `git add .`
- Only stage modified files and specified files when commanded
- No commits unless explicitly commanded
- Generate concise commit messages from `git diff HEAD`

## Development Setup
- Use `.github/setup-dev.sh` for environment setup
- Pre-commit hook enforces formatting and clippy checks
- All changes must pass CI pipeline
