// src/data_input/bbl_reader.rs

use std::error::Error;
use std::path::{Path, PathBuf};

use bbl_parser::{export_to_csv, parse_bbl_file_all_logs, ExportOptions};

use crate::types::BblExpansionResult;

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
/// `scratch_dir` is `Some(dir)` — the exact directory the caller must remove afterward — only
/// when `keep` is false; it is the directory actually passed to `bbl_parser`, tracked explicitly
/// rather than re-derived from a CSV path's `.parent()` (fragile: depends on bbl_parser never
/// nesting its output in a subdirectory, and silently does nothing if a path ever lacks a
/// parent). `None` when `keep` is true: flight CSVs are written directly to `keep_base_dir` (the
/// resolved `--output-dir`, or the source `.bbl`'s own folder when that's not given) and the
/// caller must never remove them — see `--keep` in `main.rs`.
pub fn expand_bbl_to_scratch_csvs(
    bbl_path: &Path,
    force_export: bool,
    keep: bool,
    keep_base_dir: Option<&Path>,
    debug_mode: bool,
) -> Result<BblExpansionResult, Box<dyn Error>> {
    let origin_dir = bbl_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    let export_dir = if keep {
        keep_base_dir
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|| origin_dir.clone())
    } else {
        static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        std::env::temp_dir().join(format!("bbcsvr-bbl-{}-{unique_id}", std::process::id()))
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
        let Some(csv_path) = report.csv_path else {
            eprintln!(
                "⚠️  Skipping flight {} in {}: bbl_parser produced no CSV path",
                log.log_number,
                bbl_path.display()
            );
            skipped_count += 1;
            continue;
        };
        results.push((csv_path.to_string_lossy().to_string(), origin_dir.clone()));
    }

    if results.is_empty() && skipped_count > 0 {
        eprintln!(
            "⚠️  All {skipped_count} flight(s) in {} were filtered or failed to export (use -F/--force-export for filtered flights)",
            bbl_path.display()
        );
    }

    let scratch_dir = if keep { None } else { Some(export_dir) };
    Ok((results, scratch_dir))
}

/// Removes a scratch directory returned by `expand_bbl_to_scratch_csvs`. Best-effort: called
/// after all scratch CSVs have been parsed, at end of program. Takes the scratch directory
/// itself (tracked explicitly by the caller), never a CSV file path — removing a directory
/// inferred from an arbitrary path is exactly the kind of mistake this function must not permit.
pub fn cleanup_scratch_dir(scratch_dir: &Path) {
    let _ = std::fs::remove_dir_all(scratch_dir);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_file_returns_err_not_panic() {
        let result = expand_bbl_to_scratch_csvs(
            Path::new("/nonexistent/path/flight.BBL"),
            false,
            false,
            None,
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

        let result = expand_bbl_to_scratch_csvs(&bbl_path, false, false, None, false);
        assert!(
            result.is_err(),
            "garbage input must error, not panic or silently succeed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

// src/data_input/bbl_reader.rs
