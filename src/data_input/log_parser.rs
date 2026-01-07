// src/data_input/log_parser.rs

use csv::ReaderBuilder;
use std::error::Error;
use std::fs::File;
use std::io::{BufRead, BufReader, Seek, SeekFrom};
use std::path::Path;

use crate::data_input::log_data::LogRowData;
use crate::types::LogParseResult;

/// Reads a separate .headers.csv file and extracts header metadata key-value pairs
fn read_headers_csv(headers_file_path: &Path) -> Result<Vec<(String, String)>, Box<dyn Error>> {
    let mut header_metadata = Vec::new();
    let file = File::open(headers_file_path)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(false) // Headers files typically don't have a header row
        .trim(csv::Trim::All)
        .from_reader(BufReader::new(file));

    for result in reader.records() {
        match result {
            Ok(record) => {
                if record.len() >= 2 {
                    let key = record
                        .get(0)
                        .unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .to_string();
                    let value = record
                        .get(1)
                        .unwrap_or("")
                        .trim()
                        .trim_matches('"')
                        .to_string();
                    if !key.is_empty() {
                        header_metadata.push((key, value));
                    }
                }
            }
            Err(e) => {
                println!("Warning: Error parsing headers CSV line: {e}");
                // Continue processing other lines
            }
        }
    }

    Ok(header_metadata)
}

/// Parses the CSV log file, extracts data, determines header presence, and calculates sample rate.
///
/// Returns a tuple containing:
/// 1. `Vec<LogRowData>`: All parsed log data rows.
/// 2. `Option<f64>`: The estimated sample rate in Hz.
/// 3. `[bool; 3]`: Flags indicating if f_term[0-2] headers were found.
/// 4. `[bool; 4]`: Flags indicating if setpoint[0-3] headers were found.
/// 5. `[bool; 3]`: Flags indicating if gyroADC[0-2] headers were found.
/// 6. `[bool; 3]`: Flags indicating if gyroUnfilt[0-2] headers were found.
/// 7. `[bool; 4]`: Flags indicating if debug[0-3] headers were found.
/// 8. `Vec<(String, String)>`: Header metadata key-value pairs found before CSV headers.
pub fn parse_log_file(input_file_path: &Path, debug_mode: bool) -> LogParseResult {
    // --- Header Definition and Index Mapping ---
    let target_headers = [
        "time (us)", // 0
        "axisP[0]",
        "axisP[1]",
        "axisP[2]", // 1, 2, 3
        "axisI[0]",
        "axisI[1]",
        "axisI[2]", // 4, 5, 6
        "axisD[0]",
        "axisD[1]",
        "axisD[2]", // 7, 8, 9
        "axisF[0]",
        "axisF[1]",
        "axisF[2]", // 10, 11, 12
        "setpoint[0]",
        "setpoint[1]",
        "setpoint[2]",
        "setpoint[3]", // 13, 14, 15, 16 (setpoint[3] is throttle)
        "gyroADC[0]",
        "gyroADC[1]",
        "gyroADC[2]", // 17, 18, 19
        "gyroUnfilt[0]",
        "gyroUnfilt[1]",
        "gyroUnfilt[2]", // 20, 21, 22
        "debug[0]",
        "debug[1]",
        "debug[2]",
        "debug[3]", // 23, 24, 25, 26
    ];

    // --- Header Metadata Extraction and CSV Position Tracking ---
    let mut header_metadata: Vec<(String, String)> = Vec::new();

    // First, check for separate .headers.csv file (Type 1)
    let headers_file_path = {
        let file_stem = input_file_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("");
        input_file_path.with_file_name(format!("{file_stem}.headers.csv"))
    };

    if headers_file_path.exists() {
        if debug_mode {
            println!("Found separate headers file: {headers_file_path:?}");
        }
        match read_headers_csv(&headers_file_path) {
            Ok(mut headers_metadata) => {
                if debug_mode {
                    println!(
                        "Successfully read {} entries from headers file",
                        headers_metadata.len()
                    );
                }
                header_metadata.append(&mut headers_metadata);
            }
            Err(e) => {
                println!("Warning: Failed to read headers file: {e}");
            }
        }
    }

    let csv_start_position: u64;

    // Next, extract embedded header metadata from the CSV file itself (Type 2)
    // Single-pass file reading: extract header metadata and find CSV start position
    let embedded_start_count = header_metadata.len(); // Track how many we had from headers file
    {
        let mut file = File::open(input_file_path)?;
        let mut reader = BufReader::new(&mut file);
        let mut line_buffer = String::new();
        let mut current_position = 0u64;

        loop {
            line_buffer.clear();
            let bytes_read = reader.read_line(&mut line_buffer)?;
            if bytes_read == 0 {
                return Err("Reached end of file without finding CSV headers".into());
            }

            let trimmed_line = line_buffer.trim();

            // Skip empty lines
            if trimmed_line.is_empty() {
                current_position += bytes_read as u64;
                continue;
            }

            // Check if this line contains the CSV headers
            if trimmed_line.contains("time")
                && (trimmed_line.contains("axisP") || trimmed_line.contains("gyroADC"))
            {
                csv_start_position = current_position;
                if debug_mode {
                    println!("Found CSV headers at file position {csv_start_position}");
                }
                break;
            }

            // Parse metadata line directly without CSV reader (more efficient)
            if trimmed_line.contains(',') {
                let parts: Vec<&str> = trimmed_line.splitn(2, ',').collect();
                if parts.len() == 2 {
                    let key = parts[0].trim().trim_matches('"').to_string();
                    let value = parts[1].trim().trim_matches('"').to_string();
                    if !key.is_empty() {
                        header_metadata.push((key, value));
                    }
                }
            }

            current_position += bytes_read as u64;
        }
    }

    if embedded_start_count > 0 {
        println!(
            "Extracted {} total header metadata (separate header file)",
            header_metadata.len()
        );
    } else {
        println!(
            "Extracted {} total header metadata (embedded)",
            header_metadata.len()
        );
    }
    if debug_mode && !header_metadata.is_empty() {
        println!("Sample header metadata:");
        for (i, (key, value)) in header_metadata.iter().take(5).enumerate() {
            println!("  {}: '{}' = '{}'", i + 1, key, value);
        }
        if header_metadata.len() > 5 {
            println!("  ... and {} more", header_metadata.len() - 5);
        }
    }

    let mut setpoint_header_found = [false; 4];
    let mut gyro_header_found = [false; 3];
    let mut gyro_unfilt_header_found = [false; 3];
    let mut debug_header_found = [false; 4];
    let mut f_term_header_found = [false; 3];

    let header_indices: Vec<Option<usize>>;
    let mut motor_indices: Vec<usize> = Vec::new();

    // Read CSV header and map target headers to indices.
    {
        let mut file = File::open(input_file_path)?;
        file.seek(SeekFrom::Start(csv_start_position))?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));
        let header_record = reader.headers()?.clone();
        println!("Flight data keys found in CSV: {header_record:?}");

        // Detect motor channels dynamically (motor[0] through motor[N-1])
        // Collect all motor headers with their (motor_num, csv_idx) pairs, allowing out-of-order
        let mut motor_pairs: Vec<(usize, usize)> = Vec::new();

        for (csv_idx, header) in header_record.iter().enumerate() {
            let trimmed = header.trim();
            if trimmed.starts_with("motor[") && trimmed.ends_with(']') {
                if let Some(num_str) = trimmed
                    .strip_prefix("motor[")
                    .and_then(|s| s.strip_suffix(']'))
                {
                    if let Ok(motor_num) = num_str.parse::<usize>() {
                        motor_pairs.push((motor_num, csv_idx));
                    }
                }
            }
        }

        // Sort by motor number to ensure consistent ordering
        motor_pairs.sort_by_key(|&(motor_num, _)| motor_num);

        // Validate sequence and collect any missing motor indices
        if !motor_pairs.is_empty() {
            let mut missing_indices: Vec<usize> = Vec::new();
            let mut expected_motor = 0usize;

            for &(motor_num, _) in &motor_pairs {
                // Collect all missing indices between expected and found
                while expected_motor < motor_num {
                    missing_indices.push(expected_motor);
                    expected_motor += 1;
                }
                expected_motor = motor_num + 1;
            }

            // Emit single consolidated warning if debug_mode and gaps detected
            if debug_mode && !missing_indices.is_empty() {
                let missing_str = missing_indices
                    .iter()
                    .map(|i| i.to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                println!(
                    "Warning: Gap(s) detected in motor indices. Missing: motor[{}]",
                    missing_str
                );
            }

            // Populate motor_indices from sorted pairs
            for (_, csv_idx) in &motor_pairs {
                motor_indices.push(*csv_idx);
            }

            println!(
                "Detected {} motor outputs: motor[{}] through motor[{}]",
                motor_indices.len(),
                motor_pairs.first().map(|&(n, _)| n).unwrap_or(0),
                motor_pairs.last().map(|&(n, _)| n).unwrap_or(0)
            );
        }

        header_indices = target_headers
            .iter()
            .enumerate()
            .map(|(i, &target_header)| {
                if i == 0 {
                    // Special case for time header: check for both "time (us)" and "time"
                    header_record.iter().position(|h| {
                        let trimmed = h.trim();
                        trimmed == "time (us)" || trimmed == "time"
                    })
                } else {
                    header_record.iter().position(|h| h.trim() == target_header)
                }
            })
            .collect();

        println!("Flight data key mapping:");
        let mut essential_pid_headers_found = true;

        // Check essential PID headers (Time, P, I, D[0], D[1]).
        for i in 0..=8 {
            // Indices 0 through 8
            let name = target_headers[i];
            let found = header_indices[i].is_some();
            println!(
                "  '{}': {}",
                name,
                if found { "Found" } else { "Not Found" }
            );
            if !found {
                essential_pid_headers_found = false;
            }
        }

        // Check optional 'axisD[2]' header (Index 9).
        let axis_d2_found_in_csv = header_indices[9].is_some();
        println!(
            "  '{}': {} (Optional, defaults to 0.0 if not found)",
            target_headers[9],
            if axis_d2_found_in_csv {
                "Found"
            } else {
                "Not Found"
            }
        );

        // Check f_term headers (Indices 10-12).
        for axis in 0..crate::axis_names::AXIS_NAMES.len() {
            f_term_header_found[axis] = header_indices[10 + axis].is_some();
            println!(
                "  '{}': {} (Optional, defaults to 0.0 if not found)",
                target_headers[10 + axis],
                if f_term_header_found[axis] {
                    "Found"
                } else {
                    "Not Found"
                }
            );
        }

        // Check setpoint headers (Indices 13-16).
        for axis in 0..4 {
            // Check setpoint[0] to setpoint[3]
            setpoint_header_found[axis] = header_indices[13 + axis].is_some();
            let purpose = if axis < 3 {
                format!("Essential for Setpoint plots and Step Response Axis {axis}")
            } else {
                "Throttle (setpoint[3])".to_string()
            };
            println!(
                "  '{}': {} ({})",
                target_headers[13 + axis],
                if setpoint_header_found[axis] {
                    "Found"
                } else {
                    "Not Found"
                },
                purpose
            );
        }

        // Check gyro (filtered) headers (Indices 17-19).
        for axis in 0..crate::axis_names::AXIS_NAMES.len() {
            gyro_header_found[axis] = header_indices[17 + axis].is_some();
            println!(
                "  '{}': {} (Essential for Step Response, Gyro plots, and PID Error Axis {})",
                target_headers[17 + axis],
                if gyro_header_found[axis] {
                    "Found"
                } else {
                    "Not Found"
                },
                axis
            );
        }

        // Check gyroUnfilt headers (Indices 20-22).
        for axis in 0..crate::axis_names::AXIS_NAMES.len() {
            gyro_unfilt_header_found[axis] = header_indices[20 + axis].is_some();
            println!(
                "  '{}': {} (Fallback for Gyro vs Unfilt Axis {})",
                target_headers[20 + axis],
                if gyro_unfilt_header_found[axis] {
                    "Found"
                } else {
                    "Not Found"
                },
                axis
            );
        }

        // Check debug headers (Indices 23-26).
        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[23 + idx_offset].is_some();
            let purpose = if idx_offset < 3 {
                "Fallback for gyroUnfilt[0-2]"
            } else {
                "Optional debug channel"
            };
            println!(
                "  '{}': {} ({})",
                target_headers[23 + idx_offset],
                if debug_header_found[idx_offset] {
                    "Found"
                } else {
                    "Not Found"
                },
                purpose
            );
        }

        if !essential_pid_headers_found {
            let missing_essentials: Vec<String> = (0..=8)
                .filter(|&i| header_indices[i].is_none())
                .map(|i| format!("'{}'", target_headers[i]))
                .collect();
            return Err(format!(
                "Error: Missing essential headers for PIDsum calculation: {}. Aborting.",
                missing_essentials.join(", ")
            )
            .into());
        }

        // Report debug fallback usage if applicable
        let using_debug_fallback = !gyro_unfilt_header_found.iter().any(|&found| found)
            && debug_header_found.iter().take(3).any(|&found| found);

        if using_debug_fallback {
            println!("  ⚠️  Using debug[0-2] as fallback for gyroUnfilt[0-2]");

            // Try to report which debug mode is being used
            if let Some((_, debug_mode_value)) =
                header_metadata.iter().find(|(k, _)| k == "debug_mode")
            {
                if let Ok(debug_int) = debug_mode_value.parse::<u32>() {
                    println!("  Debug mode value: {}", debug_int);
                }
            }
        }
    }

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading data rows...");
    {
        let mut file = File::open(input_file_path)?;
        file.seek(SeekFrom::Start(csv_start_position))?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices
                            .get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // Parse Time (us)
                    let time_us = parse_f64_by_target_idx(0); // Index 0
                    if let Some(t_us) = time_us {
                        current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                        eprintln!(
                            "Warning: Skipping row {} due to missing or invalid 'time (us)'",
                            row_index + 1
                        );
                        continue;
                    }

                    // Parse P, I, D, F, Gyro (for axes 0-2)
                    #[allow(clippy::needless_range_loop)]
                    for axis in 0..crate::axis_names::AXIS_NAMES.len() {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis); // Indices 1,2,3
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis); // Indices 4,5,6

                        // D term with optional axisD[2] fallback
                        let d_target_idx = 7 + axis; // Indices 7,8,9
                        let d_val = parse_f64_by_target_idx(d_target_idx);
                        current_row_data.d_term[axis] = if d_val.is_none() && axis == 2 {
                            Some(0.0)
                        } else {
                            d_val
                        };

                        // F term with optional axisF[0-2] fallback
                        let f_target_idx = 10 + axis; // Indices 10,11,12
                        if f_term_header_found[axis] {
                            current_row_data.f_term[axis] = parse_f64_by_target_idx(f_target_idx);
                        } else {
                            current_row_data.f_term[axis] = Some(0.0);
                        }
                        current_row_data.gyro[axis] = parse_f64_by_target_idx(17 + axis);
                        // Indices 17,18,19
                    }

                    // Parse Setpoint (for axes 0-3)
                    for axis in 0..4 {
                        current_row_data.setpoint[axis] = parse_f64_by_target_idx(13 + axis);
                        // Indices 13,14,15,16
                    }

                    // Parse gyroUnfilt and debug
                    let mut parsed_gyro_unfilt = [None; 3];
                    let mut parsed_debug = [None; 4];
                    for axis in 0..crate::axis_names::AXIS_NAMES.len() {
                        if gyro_unfilt_header_found[axis] {
                            parsed_gyro_unfilt[axis] = parse_f64_by_target_idx(20 + axis);
                            // Indices 20,21,22
                        }
                    }

                    for idx_offset in 0..4 {
                        if debug_header_found[idx_offset] {
                            parsed_debug[idx_offset] = parse_f64_by_target_idx(23 + idx_offset);
                            // Indices 23,24,25,26
                        }
                        current_row_data.debug[idx_offset] = parsed_debug[idx_offset];
                    }

                    // Apply Fallback Logic for gyro_unfilt (debug[0-2] --> gyroUnfilt[0-2])
                    for axis in 0..crate::axis_names::AXIS_NAMES.len() {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => parsed_debug[axis],
                        };
                    }

                    // Parse motor outputs
                    current_row_data.motors = Vec::with_capacity(motor_indices.len());
                    for &motor_csv_idx in &motor_indices {
                        let motor_val = record
                            .get(motor_csv_idx)
                            .and_then(|val_str| val_str.parse::<f64>().ok());
                        current_row_data.motors.push(motor_val);
                    }

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!(
                        "Warning: Skipping row {} due to CSV read error: {}",
                        row_index + 1,
                        e
                    );
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
            if let Some(sr) = sample_rate {
                println!("Estimated Sample Rate: {:.2} Hz", sr);
            }
        }
    }
    if sample_rate.is_none() {
        println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might be affected.");
    }

    Ok((
        all_log_data,
        sample_rate,
        f_term_header_found,
        setpoint_header_found,
        gyro_header_found,
        gyro_unfilt_header_found,
        debug_header_found,
        header_metadata,
    ))
}

// src/data_input/log_parser.rs
