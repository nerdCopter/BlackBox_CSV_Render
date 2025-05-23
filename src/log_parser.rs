use csv::ReaderBuilder;
use std::error::Error;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use crate::log_data::LogRowData;

/// Parses the CSV log file, extracts data, determines header presence, and calculates sample rate.
///
/// Returns a tuple containing:
/// 1. `Vec<LogRowData>`: All parsed log data rows.
/// 2. `Option<f64>`: The estimated sample rate in Hz.
/// 3. `[bool; 4]`: Flags indicating if setpoint[0-3] headers were found.
/// 4. `[bool; 3]`: Flags indicating if gyroADC[0-2] headers were found.
/// 5. `[bool; 3]`: Flags indicating if gyroUnfilt[0-2] headers were found.
/// 6. `[bool; 4]`: Flags indicating if debug[0-3] headers were found.
pub fn parse_log_file(
    input_file_path: &Path,
) -> Result<(Vec<LogRowData>, Option<f64>, [bool; 4], [bool; 3], [bool; 3], [bool; 4]), Box<dyn Error>> {
    // --- Header Definition and Index Mapping ---
    let target_headers = [
        "time (us)",            // 0
        "axisP[0]", "axisP[1]", "axisP[2]", // 1, 2, 3
        "axisI[0]", "axisI[1]", "axisI[2]", // 4, 5, 6
        "axisD[0]", "axisD[1]", "axisD[2]", // 7, 8, 9
        "setpoint[0]", "setpoint[1]", "setpoint[2]", "setpoint[3]", // 10, 11, 12, 13 (setpoint[3] is throttle)
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]", // 14, 15, 16
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]", // 17, 18, 19
        "debug[0]", "debug[1]", "debug[2]", "debug[3]", // 20, 21, 22, 23
    ];

    let mut setpoint_header_found = [false; 4];
    let mut gyro_header_found = [false; 3];
    let mut gyro_unfilt_header_found = [false; 3];
    let mut debug_header_found = [false; 4];

    let header_indices: Vec<Option<usize>>;

    // Read CSV header and map target headers to indices.
    {
        let file = File::open(input_file_path)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));
        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        header_indices = target_headers.iter().map(|&target_header| {
            header_record.iter().position(|h| h.trim() == target_header)
        }).collect();

        println!("Header mapping status:");
        let mut essential_pid_headers_found = true;

        // Check essential PID headers (Time, P, I, D[0], D[1]).
        for i in 0..=8 { // Indices 0 through 8
            let name = target_headers[i];
            let found = header_indices[i].is_some();
            println!("  '{}': {}", name, if found { "Found" } else { "Not Found" });
            if !found {
                essential_pid_headers_found = false;
            }
        }

        // Check optional 'axisD[2]' header (Index 9).
        let axis_d2_found_in_csv = header_indices[9].is_some();
        println!("  '{}': {} (Optional, defaults to 0.0 if not found)", target_headers[9], if axis_d2_found_in_csv { "Found" } else { "Not Found" });

        // Check setpoint headers (Indices 10-13).
        for axis in 0..4 { // Check setpoint[0] to setpoint[3]
            setpoint_header_found[axis] = header_indices[10 + axis].is_some();
            let purpose = if axis < 3 {
                format!("Essential for Setpoint plots and Step Response Axis {}", axis)
            } else {
                "Throttle (setpoint[3])".to_string()
            };
            println!("  '{}': {} ({})", target_headers[10 + axis], if setpoint_header_found[axis] { "Found" } else { "Not Found" }, purpose);
        }

        // Check gyro (filtered) headers (Indices 14-16).
        for axis in 0..3 {
            gyro_header_found[axis] = header_indices[14 + axis].is_some();
            println!("  '{}': {} (Essential for Step Response, Gyro plots, and PID Error Axis {})", target_headers[14 + axis], if gyro_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check gyroUnfilt headers (Indices 17-19).
        for axis in 0..3 {
            gyro_unfilt_header_found[axis] = header_indices[17 + axis].is_some();
            println!("  '{}': {} (Fallback for Gyro vs Unfilt Axis {})", target_headers[17 + axis], if gyro_unfilt_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check debug headers (Indices 20-23).
        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[20 + idx_offset].is_some();
            println!("  '{}': {} (Fallback for gyroUnfilt[0-2])", target_headers[20 + idx_offset], if debug_header_found[idx_offset] { "Found" } else { "Not Found" });
        }

        if !essential_pid_headers_found {
            let missing_essentials: Vec<String> = (0..=8).filter(|&i| header_indices[i].is_none()).map(|i| format!("'{}'", target_headers[i])).collect();
            return Err(format!("Error: Missing essential headers for PIDsum calculation: {}. Aborting.", missing_essentials.join(", ")).into());
        }
    }

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading data rows...");
    {
        let file = File::open(input_file_path)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // Parse Time (us)
                    let time_us = parse_f64_by_target_idx(0); // Index 0
                    if let Some(t_us) = time_us {
                        current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                        eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                        continue;
                    }

                    // Parse P, I, D, Gyro (for axes 0-2)
                    for axis in 0..3 {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis); // Indices 1,2,3
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis); // Indices 4,5,6

                        // D term with optional axisD[2] fallback
                        let d_target_idx = 7 + axis; // Indices 7,8,9
                        if axis == 2 && header_indices[d_target_idx].is_none() {
                            current_row_data.d_term[axis] = Some(0.0);
                        } else {
                            current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }
                        current_row_data.gyro[axis] = parse_f64_by_target_idx(14 + axis); // Indices 14,15,16
                    }

                    // Parse Setpoint (for axes 0-3)
                    for axis in 0..4 {
                        current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis); // Indices 10,11,12,13
                    }

                    // Parse gyroUnfilt and debug
                    let mut parsed_gyro_unfilt = [None; 3];
                    let mut parsed_debug = [None; 4];

                    for axis in 0..3 {
                        if gyro_unfilt_header_found[axis] {
                            parsed_gyro_unfilt[axis] = parse_f64_by_target_idx(17 + axis); // Indices 17,18,19
                        }
                    }

                    for idx_offset in 0..4 {
                        if debug_header_found[idx_offset] {
                            parsed_debug[idx_offset] = parse_f64_by_target_idx(20 + idx_offset); // Indices 20,21,22,23
                        }
                        current_row_data.debug[idx_offset] = parsed_debug[idx_offset];
                    }

                    // Apply Fallback Logic for gyro_unfilt (debug[0-2] --> gyroUnfilt[0-2])
                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => match parsed_debug[axis] {
                                Some(val) => Some(val),
                                None => None,
                            }
                        };
                    }

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    }

    println!("Finished reading {} data rows.", all_log_data.len());

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        let mut prev_time: Option<f64> = None;
        for row in &all_log_data {
            if let Some(current_time) = row.time_sec {
                if let Some(pt) = prev_time {
                    let delta = current_time - pt;
                    if delta > 1e-9 {
                        total_delta += delta;
                        count += 1;
                    }
                }
                prev_time = Some(current_time);
            }
        }
        if count > 0 {
            let avg_delta = total_delta / count as f64;
            sample_rate = Some(1.0 / avg_delta);
            println!("Estimated Sample Rate: {:.2} Hz", sample_rate.unwrap());
        }
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might be affected.");
    }

    Ok((all_log_data, sample_rate, setpoint_header_found, gyro_header_found, gyro_unfilt_header_found, debug_header_found))
}

// src/log_parser.rs
