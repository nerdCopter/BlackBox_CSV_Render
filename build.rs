use anyhow::Result;
use std::path::{Path, PathBuf};
use vergen::EmitBuilder;

/// vergen's own `rerun-if-changed` only watches the plain `HEAD` file. For a branch checkout that
/// file never changes content across commits (it stays a symbolic ref, `ref: refs/heads/<name>`;
/// only a detached-HEAD checkout rewrites it with a raw SHA). The actual per-commit-changing file
/// is the reflog (`logs/HEAD`), which gets a new line on every commit regardless of branch vs.
/// detached state. Without watching it, cargo correctly (by its own tracking) never reruns this
/// build script after the first commit on a branch, silently freezing the embedded
/// `VERGEN_GIT_SHA` at whatever it was on that first build. Confirmed directly: multiple commits
/// landed on `feat/113-direct-bbl-read` with `--version` still printing a SHA from several
/// commits earlier, until a `touch build.rs` forced a rerun.
fn resolve_git_dir() -> Option<PathBuf> {
    let dot_git = Path::new(".git");
    if dot_git.is_dir() {
        return Some(dot_git.to_path_buf());
    }
    // A worktree's `.git` is a file: "gitdir: /path/to/main/repo/.git/worktrees/<name>"
    let contents = std::fs::read_to_string(dot_git).ok()?;
    let path_str = contents.strip_prefix("gitdir:")?.trim();
    Some(PathBuf::from(path_str))
}

fn main() -> Result<()> {
    if let Some(git_dir) = resolve_git_dir() {
        println!(
            "cargo:rerun-if-changed={}",
            git_dir.join("logs/HEAD").display()
        );
    }
    EmitBuilder::builder()
        .git_sha(true)
        .git_commit_date()
        .emit()?;
    Ok(())
}
