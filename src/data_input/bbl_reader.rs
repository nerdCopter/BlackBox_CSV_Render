// src/data_input/bbl_reader.rs

use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::error::Error;
use std::hash::{Hash, Hasher};
use std::path::{Path, PathBuf};

use bbl_parser::{export_to_csv, parse_bbl_file_all_logs, ExportOptions};

use crate::types::BblExpansionResult;

/// 6 hex chars (24 bits) of `DefaultHasher` over `path`'s canonicalized form (falling back to the
/// given path if canonicalization fails, e.g. a broken symlink) — case-sensitive, so
/// `flight.bbl`/`flight.BBL` in the same directory hash differently too, not just cross-directory
/// same-basename inputs. Collision probability only matters within one colliding-stem group (a
/// handful of files in realistic use, see IT #197), where 24 bits is a comfortable margin;
/// `rename_unique` below is the defensive backstop regardless.
fn path_disambiguator(path: &Path) -> String {
    let hash_input = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut hasher = DefaultHasher::new();
    hash_input.hash(&mut hasher);
    format!("{:06x}", hasher.finish() & 0xFF_FFFF)
}

/// Renames `path` to `new_filename` in the same directory. No fixed-length hash is mathematically
/// collision-proof, only collision-*unlikely* — if the computed target already exists (a true
/// hash collision, or an unrelated file that happens to already be there), fall back to an
/// incrementing counter rather than silently overwriting it.
fn rename_unique(path: &Path, new_filename: &str) -> Result<PathBuf, Box<dyn Error>> {
    let mut target = path.with_file_name(new_filename);
    if target.exists() {
        let stem = Path::new(new_filename)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let ext = Path::new(new_filename)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("csv");
        let mut n = 2;
        loop {
            target = path.with_file_name(format!("{stem}-{n}.{ext}"));
            if !target.exists() {
                break;
            }
            n += 1;
        }
    }
    std::fs::rename(path, &target)?;
    Ok(target)
}

/// Decodes a `.bbl`/`.BBL` file via the `bbl_parser` crate and exports each embedded flight to a
/// scratch CSV/`.headers.csv` pair in the OS temp directory, reusing `bbl_parser`'s own CSV
/// export (its documented, stable output format) instead of depending on `DecodedFrame.data`'s
/// internal field-name keys. The scratch files feed the existing `log_parser::parse_log_file`
/// pipeline unchanged, so motor/eRPM channel alignment and header-detection logic are not
/// duplicated.
///
/// `bbl_parser`'s own low-value-flight heuristic (too short, low data density, or minimal gyro
/// activity — ground tests/arm checks) is applied by `export_to_csv` itself (bbl_parser >=
/// 1.1.0), gated by `ExportOptions.force_export` (`-F`/`--force-export`) — a skipped flight
/// returns an `ExportReport` with `csv_path: None` and `skip_reason: Some(reason)` instead of
/// writing files. No separate pre-check needed here.
///
/// Returns `(flights, scratch_dir)`. `flights` has one `(scratch_csv_path,
/// original_bbl_parent_dir)` pair per flight that was exported (flights skipped by the
/// heuristic, or that individually failed to export, are simply absent, not a whole-file error —
/// one bad flight in a multi-flight file must not discard the flights that already exported
/// fine), in flight order. `original_bbl_parent_dir` lets the caller default `--output-dir` to
/// the source `.bbl`'s own folder rather than the scratch directory.
///
/// `scratch_dir` is `Some(guard)` — an exclusively-created temp directory that removes itself
/// (and everything under it) when dropped — only when `keep` is false. The caller must keep the
/// guard alive until it is done reading the flight CSVs, then drop it explicitly (or let it fall
/// out of scope) to clean up. Using `tempfile::TempDir` instead of a predictable
/// `pid`-plus-counter path avoids a real collision risk: a stale directory left behind by an
/// earlier crashed run with a reused PID, or another local process racing to the same path,
/// could otherwise be silently reused and then `remove_dir_all`'d by a run that never created it.
/// `None` when `keep` is true: flight CSVs are written directly to `keep_base_dir` (the resolved
/// `--output-dir`, or the source `.bbl`'s own folder when that's not given) and must never be
/// removed — see `--keep` in `main.rs`.
pub fn expand_bbl_to_scratch_csvs(
    bbl_path: &Path,
    force_export: bool,
    keep: bool,
    keep_base_dir: Option<&Path>,
    colliding_stems: &HashSet<String>,
    debug_mode: bool,
) -> Result<BblExpansionResult, Box<dyn Error>> {
    let origin_dir = bbl_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    // Held for the whole function so its directory survives until export_to_csv has written
    // into it; None under --keep, where flights go straight to their real, permanent location.
    let scratch_guard = if keep {
        None
    } else {
        Some(
            tempfile::Builder::new()
                .prefix("bbcsvr-bbl-")
                .tempdir()
                .map_err(|e| format!("failed to create scratch directory: {e}"))?,
        )
    };

    let export_dir: PathBuf = match (&scratch_guard, keep) {
        (Some(guard), _) => guard.path().to_path_buf(),
        (None, true) => keep_base_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| origin_dir.clone()),
        (None, false) => unreachable!("scratch_guard is only None when keep is true"),
    };

    let export_options = ExportOptions {
        csv: true,
        gpx: false,
        event: false,
        output_dir: Some(export_dir.to_string_lossy().to_string()),
        force_export,
    };

    if keep {
        println!(
            "Exporting {} (BBL) to {}...",
            bbl_path.display(),
            export_dir.display()
        );
    } else {
        println!("Exporting {} (BBL) to scratch CSV...", bbl_path.display());
    }
    // Callers (expand_one_bbl_file) already print bbl_path alongside any Err returned here, so
    // these messages carry only the cause — not the path again — to avoid double-printing it.
    let logs = parse_bbl_file_all_logs(bbl_path, export_options.clone(), debug_mode)
        .map_err(|e| format!("failed to parse: {e}"))?;

    if logs.is_empty() {
        return Err("no flight logs found".into());
    }

    let mut results = Vec::with_capacity(logs.len());
    let mut skipped_count = 0;
    for log in &logs {
        // A per-flight export failure (disk full, permissions) must not discard flights that
        // already exported successfully earlier in this same .bbl — report and skip, don't ?.
        let report = match export_to_csv(log, bbl_path, &export_options, None) {
            Ok(report) => report,
            Err(e) => {
                eprintln!(
                    "⚠️  Skipping flight {} in {}: export failed: {e}",
                    log.log_number,
                    bbl_path.display()
                );
                skipped_count += 1;
                continue;
            }
        };
        if let Some(reason) = report.skip_reason {
            eprintln!(
                "⚠️  Skipping flight {} in {}: {reason} (use -F/--force-export to include it)",
                log.log_number,
                bbl_path.display()
            );
            skipped_count += 1;
            continue;
        }
        let Some(mut csv_path) = report.csv_path else {
            eprintln!(
                "⚠️  Skipping flight {} in {}: bbl_parser produced no CSV path",
                log.log_number,
                bbl_path.display()
            );
            skipped_count += 1;
            continue;
        };

        // Only a retained export in a shared directory can collide with another input's export
        // (scratch/tempdir mode is always exclusive per file) — and only when this file's own
        // basename was seen more than once in the pre-scan (find_colliding_bbl_stems). Renaming
        // here, immediately after export, is required: bbl_parser's own File::create truncates
        // an existing same-named file, so a collision detected only after both exports have run
        // is already unrecoverable — the first file's content would already be gone.
        if keep {
            let stem_collides = bbl_path
                .file_stem()
                .map(|s| s.to_string_lossy().to_ascii_lowercase())
                .is_some_and(|s| colliding_stems.contains(&s));
            if stem_collides {
                let suffix = path_disambiguator(bbl_path);
                let csv_stem = csv_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("output")
                    .to_string();
                match rename_unique(&csv_path, &format!("{csv_stem}.{suffix}.csv")) {
                    Ok(renamed_csv) => {
                        // log_parser locates the headers sidecar as `{csv_file_stem}.headers.csv`
                        // (log_parser.rs), so the headers file must be renamed to match the CSV's
                        // *new* stem exactly, not just have the same suffix spliced into its own
                        // name — its own file_stem is "<...>.headers", not "<...>".
                        if let Some(headers_path) = &report.headers_path {
                            let new_stem = renamed_csv
                                .file_stem()
                                .and_then(|s| s.to_str())
                                .unwrap_or(&csv_stem);
                            if let Err(e) =
                                rename_unique(headers_path, &format!("{new_stem}.headers.csv"))
                            {
                                eprintln!(
                                    "⚠️  Flight {} in {}: renamed CSV to avoid a basename collision but failed to rename its headers file: {e}",
                                    log.log_number,
                                    bbl_path.display()
                                );
                            }
                        }
                        csv_path = renamed_csv;
                    }
                    Err(e) => eprintln!(
                        "⚠️  Flight {} in {}: kept CSV name collides with another input but could not be renamed: {e}",
                        log.log_number,
                        bbl_path.display()
                    ),
                }
            }
        }

        results.push((csv_path.to_string_lossy().to_string(), origin_dir.clone()));
    }

    if results.is_empty() && skipped_count > 0 {
        eprintln!(
            "⚠️  All {skipped_count} flight(s) in {} were filtered or failed to export (use -F/--force-export for filtered flights)",
            bbl_path.display()
        );
    }

    Ok((results, scratch_guard))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_file_returns_err_not_panic() {
        let empty_collisions = HashSet::new();
        let result = expand_bbl_to_scratch_csvs(
            Path::new("/nonexistent/path/flight.BBL"),
            false,
            false,
            None,
            &empty_collisions,
            false,
        );
        assert!(result.is_err());
    }

    #[test]
    fn garbage_content_returns_err_not_panic() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-bbl-reader-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        let bbl_path = dir.join("not_really_a_bbl.BBL");
        std::fs::write(&bbl_path, b"this is not blackbox log data")
            .expect("failed to write garbage test file");

        let empty_collisions = HashSet::new();
        let result =
            expand_bbl_to_scratch_csvs(&bbl_path, false, false, None, &empty_collisions, false);
        assert!(
            result.is_err(),
            "garbage input must error, not panic or silently succeed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn path_disambiguator_differs_for_case_only_paths() {
        // Nonexistent paths, so canonicalize() fails and the raw-path fallback is exercised —
        // this is the exact IT #197 same-directory case-only collision (flight.bbl vs flight.BBL).
        let a = path_disambiguator(Path::new("/nonexistent/dir/flight.bbl"));
        let b = path_disambiguator(Path::new("/nonexistent/dir/flight.BBL"));
        assert_ne!(
            a, b,
            "case-only path difference must produce different suffixes"
        );
        assert_eq!(a.len(), 6);
        assert!(a.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn path_disambiguator_is_deterministic() {
        let p = Path::new("/nonexistent/dir/flight.bbl");
        assert_eq!(path_disambiguator(p), path_disambiguator(p));
    }

    #[test]
    fn rename_unique_uses_natural_name_when_free() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-rename-unique-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        let src = dir.join("source.csv");
        std::fs::write(&src, b"data").expect("failed to write test file");

        let result = rename_unique(&src, "target.csv").expect("rename must succeed");
        assert_eq!(result, dir.join("target.csv"));
        assert!(result.exists());
        assert!(!src.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn rename_unique_falls_back_to_counter_on_real_collision() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-rename-collision-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        // Target already exists before the rename — a genuine (if astronomically unlikely) hash
        // collision, or an unrelated pre-existing file. Must never be silently overwritten.
        std::fs::write(dir.join("target.csv"), b"must not be touched")
            .expect("failed to write pre-existing target file");
        let src = dir.join("source.csv");
        std::fs::write(&src, b"new data").expect("failed to write test file");

        let result = rename_unique(&src, "target.csv").expect("rename must succeed via fallback");
        assert_eq!(result, dir.join("target-2.csv"));
        assert_eq!(
            std::fs::read_to_string(dir.join("target.csv")).unwrap(),
            "must not be touched",
            "the pre-existing colliding file must survive untouched"
        );
        assert_eq!(std::fs::read_to_string(&result).unwrap(), "new data");

        std::fs::remove_dir_all(&dir).ok();
    }
}

// src/data_input/bbl_reader.rs
