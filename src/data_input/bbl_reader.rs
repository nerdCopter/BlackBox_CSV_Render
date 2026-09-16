// src/data_input/bbl_reader.rs

use std::error::Error;
use std::path::{Path, PathBuf};

use bbl_parser::{export_to_csv, parse_bbl_file_all_logs, ExportOptions};

/// Decodes a `.bbl`/`.BBL` file via the `bbl_parser` crate and exports each embedded flight to a
/// scratch CSV/`.headers.csv` pair in the OS temp directory, reusing `bbl_parser`'s own CSV
/// export (its documented, stable output format) instead of depending on `DecodedFrame.data`'s
/// internal field-name keys. The scratch files feed the existing `log_parser::parse_log_file`
/// pipeline unchanged, so motor/eRPM channel alignment and header-detection logic are not
/// duplicated. `force_export` is always set: a file the user explicitly named must never be
/// silently skipped by bbl_parser's short/low-density heuristics.
///
/// Returns one `(scratch_csv_path, original_bbl_parent_dir)` pair per flight, in flight order.
/// `original_bbl_parent_dir` lets the caller default `--output-dir` to the source `.bbl`'s own
/// folder rather than the scratch directory.
pub fn expand_bbl_to_scratch_csvs(
    bbl_path: &Path,
    debug_mode: bool,
) -> Result<Vec<(String, PathBuf)>, Box<dyn Error>> {
    let origin_dir = bbl_path
        .parent()
        .map(|p| p.to_path_buf())
        .unwrap_or_else(|| PathBuf::from("."));

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    let unique_id = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let scratch_dir =
        std::env::temp_dir().join(format!("bbcsvr-bbl-{}-{unique_id}", std::process::id()));

    let export_options = ExportOptions {
        csv: true,
        gpx: false,
        event: false,
        output_dir: Some(scratch_dir.to_string_lossy().to_string()),
        force_export: true,
    };

    let logs = parse_bbl_file_all_logs(bbl_path, export_options.clone(), debug_mode)
        .map_err(|e| format!("Failed to parse BBL file '{}': {e}", bbl_path.display()))?;

    if logs.is_empty() {
        return Err(format!("No flight logs found in BBL file '{}'", bbl_path.display()).into());
    }

    let mut results = Vec::with_capacity(logs.len());
    for log in &logs {
        let report = export_to_csv(log, bbl_path, &export_options, None).map_err(|e| {
            format!(
                "Failed to export flight {} of '{}' to scratch CSV: {e}",
                log.log_number,
                bbl_path.display()
            )
        })?;
        let csv_path = report.csv_path.ok_or_else(|| {
            format!(
                "bbl_parser produced no CSV path for '{}'",
                bbl_path.display()
            )
        })?;
        results.push((csv_path.to_string_lossy().to_string(), origin_dir.clone()));
    }

    Ok(results)
}

/// Removes a scratch directory created by `expand_bbl_to_scratch_csvs`. Best-effort: called
/// after all scratch CSVs have been parsed, at end of program.
pub fn cleanup_scratch_dir(scratch_csv_path: &str) {
    if let Some(dir) = Path::new(scratch_csv_path).parent() {
        let _ = std::fs::remove_dir_all(dir);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nonexistent_file_returns_err_not_panic() {
        let result = expand_bbl_to_scratch_csvs(Path::new("/nonexistent/path/flight.BBL"), false);
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

        let result = expand_bbl_to_scratch_csvs(&bbl_path, false);
        assert!(
            result.is_err(),
            "garbage input must error, not panic or silently succeed"
        );

        std::fs::remove_dir_all(&dir).ok();
    }
}

// src/data_input/bbl_reader.rs
