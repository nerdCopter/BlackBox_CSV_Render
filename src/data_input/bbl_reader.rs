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
/// `rename_pair_unique` below is the defensive backstop regardless.
fn path_disambiguator(path: &Path) -> String {
    let hash_input = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
    let mut hasher = DefaultHasher::new();
    hash_input.hash(&mut hasher);
    format!("{:06x}", hasher.finish() & 0xFF_FFFF)
}

/// Cap on `rename_pair_unique`'s fallback counter. Not a realistic ceiling for a real collision
/// (a true hash collision or a concurrent writer occupying every candidate up to this point is
/// already astronomically unlikely) — purely a defensive bound so the loop has a guaranteed
/// termination property instead of running to `u32` overflow in a pathological case.
const RENAME_PAIR_MAX_ATTEMPTS: u32 = 10_000;

/// Renames `csv_path` (and `headers_path`, when present) to `{csv_stem}.{suffix}.csv` /
/// `{csv_stem}.{suffix}.headers.csv` in the same directory. Both target names are selected and
/// existence-checked together under one shared fallback counter — never renamed independently —
/// so a collision on only one of the two (e.g. a stale headers file already sitting at the
/// natural target) can never detach the pair. `log_parser` derives the headers sidecar as
/// `{csv_file_stem}.headers.csv` (`log_parser.rs`), so the CSV and headers filenames must share
/// exactly the same stem, not just the same suffix. No fixed-length hash is mathematically
/// collision-proof, only collision-*unlikely* — the counter fallback is the defensive backstop
/// for a true hash collision or an unrelated file already occupying the target.
///
/// If the headers rename fails after the CSV rename already succeeded (e.g. the headers file
/// disappeared, a permissions or quota error), the CSV rename is rolled back before returning
/// `Err` — the caller's own fallback keeps the pre-rename `csv_path`, which must still exist on
/// disk for that to be correct. Left half-renamed, the caller would silently be pointed at a CSV
/// path that no longer exists.
///
/// This function assumes single-process, non-concurrent use of the output directory (this tool's
/// actual usage pattern) — it does not guard the gap between the existence check and the rename
/// against another process racing to the same target.
fn rename_pair_unique(
    csv_path: &Path,
    headers_path: Option<&Path>,
    csv_stem: &str,
    suffix: &str,
) -> Result<(PathBuf, Option<PathBuf>), Box<dyn Error>> {
    let mut attempt: Option<u32> = None;
    loop {
        let tag = match attempt {
            None => suffix.to_string(),
            Some(n) => format!("{suffix}-{n}"),
        };
        let csv_target = csv_path.with_file_name(format!("{csv_stem}.{tag}.csv"));
        let headers_target =
            headers_path.map(|_| csv_path.with_file_name(format!("{csv_stem}.{tag}.headers.csv")));
        let taken = csv_target.exists() || headers_target.as_ref().is_some_and(|p| p.exists());
        if !taken {
            std::fs::rename(csv_path, &csv_target)?;
            if let (Some(hp), Some(ht)) = (headers_path, &headers_target) {
                if let Err(e) = std::fs::rename(hp, ht) {
                    // Best-effort rollback so a caller that sees Err can still trust the
                    // original csv_path — leaving the CSV renamed with no matching headers
                    // move would just be a different flavor of the detachment this fixes.
                    let _ = std::fs::rename(&csv_target, csv_path);
                    return Err(e.into());
                }
                return Ok((csv_target, Some(ht.clone())));
            }
            return Ok((csv_target, None));
        }
        let next = attempt.map_or(2, |n| n + 1);
        if next > RENAME_PAIR_MAX_ATTEMPTS {
            return Err(format!(
                "no free filename found for {csv_stem}.{suffix}.csv after {RENAME_PAIR_MAX_ATTEMPTS} attempts"
            )
            .into());
        }
        attempt = Some(next);
    }
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
                match rename_pair_unique(
                    &csv_path,
                    report.headers_path.as_deref(),
                    &csv_stem,
                    &suffix,
                ) {
                    Ok((renamed_csv, _renamed_headers)) => {
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
    fn rename_pair_unique_uses_natural_names_when_free() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-rename-pair-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        let csv = dir.join("source.csv");
        let headers = dir.join("source.headers.csv");
        std::fs::write(&csv, b"data").expect("failed to write test csv");
        std::fs::write(&headers, b"meta").expect("failed to write test headers");

        let (new_csv, new_headers) = rename_pair_unique(&csv, Some(&headers), "flight", "abc123")
            .expect("rename must succeed");
        assert_eq!(new_csv, dir.join("flight.abc123.csv"));
        assert_eq!(new_headers, Some(dir.join("flight.abc123.headers.csv")));
        assert!(new_csv.exists());
        assert!(new_headers.unwrap().exists());
        assert!(!csv.exists());
        assert!(!headers.exists());

        std::fs::remove_dir_all(&dir).ok();
    }

    /// Reproduces the exact pairing bug CodeRabbit found in this PR's first pass: only the
    /// *headers* target pre-existing (CSV target free) must not detach the pair by falling back
    /// on the headers file alone — both must move to the next shared candidate together, so
    /// `log_parser`'s `{csv_file_stem}.headers.csv` derivation still finds the right sidecar.
    #[test]
    fn rename_pair_unique_keeps_pair_aligned_when_only_headers_target_collides() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-rename-pair-collision-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        // Only the natural headers target pre-exists — the natural CSV target is free.
        std::fs::write(
            dir.join("flight.abc123.headers.csv"),
            b"must not be touched",
        )
        .expect("failed to write pre-existing headers target");
        let csv = dir.join("source.csv");
        let headers = dir.join("source.headers.csv");
        std::fs::write(&csv, b"new csv data").expect("failed to write test csv");
        std::fs::write(&headers, b"new headers data").expect("failed to write test headers");

        let (new_csv, new_headers) = rename_pair_unique(&csv, Some(&headers), "flight", "abc123")
            .expect("rename must succeed via shared fallback");

        // Both targets must share the SAME fallback suffix — the CSV must not land on its
        // natural (unoccupied) name while headers alone gets bumped, which would detach them.
        assert_eq!(new_csv, dir.join("flight.abc123-2.csv"));
        assert_eq!(new_headers, Some(dir.join("flight.abc123-2.headers.csv")));
        assert_eq!(
            std::fs::read_to_string(dir.join("flight.abc123.headers.csv")).unwrap(),
            "must not be touched"
        );
        assert_eq!(
            std::fs::read_to_string(&new_csv).unwrap(),
            "new csv data",
            "CSV must move with its headers file, not stay at the natural (now-misaligned) name"
        );
        assert_eq!(
            std::fs::read_to_string(new_headers.unwrap()).unwrap(),
            "new headers data"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    /// If the headers rename fails after the CSV rename already succeeded, the CSV must be
    /// rolled back — otherwise the caller's `Err` branch would keep pointing at a `csv_path`
    /// that no longer exists on disk (a different flavor of the detachment this whole function
    /// exists to prevent). Simulated by passing a headers path that doesn't actually exist, so
    /// its `fs::rename` call fails deterministically.
    #[test]
    fn rename_pair_unique_rolls_back_csv_when_headers_rename_fails() {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!(
            "bbcsvr-rename-pair-rollback-test-{}-{unique_id}",
            std::process::id()
        ));
        std::fs::create_dir_all(&dir).expect("failed to create test dir");
        let csv = dir.join("source.csv");
        std::fs::write(&csv, b"csv data").expect("failed to write test csv");
        let missing_headers = dir.join("does_not_exist.headers.csv");

        let result = rename_pair_unique(&csv, Some(&missing_headers), "flight", "abc123");
        assert!(
            result.is_err(),
            "must fail when the headers rename fails, not silently succeed"
        );
        assert!(
            csv.exists(),
            "CSV must be rolled back to its original path so the caller's csv_path is still valid"
        );
        assert!(!dir.join("flight.abc123.csv").exists());
        assert_eq!(std::fs::read_to_string(&csv).unwrap(), "csv data");

        std::fs::remove_dir_all(&dir).ok();
    }
}

// src/data_input/bbl_reader.rs
