use anyhow::Result;
use vergen::EmitBuilder;

fn main() -> Result<()> {
    EmitBuilder::builder()
        .git_sha(true)
        .git_commit_date()
        .emit()?;
    Ok(())
}
