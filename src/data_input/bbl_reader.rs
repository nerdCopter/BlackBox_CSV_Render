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

    Ok((results, scratch_guard))
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
