// src/main.rs

mod log_data;
mod constants;
mod plotting_utils;
mod step_response;

use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

use ndarray::s; // Import the s! macro for slicing
use ndarray::{Array1, Array2}; // Import Array2
use ndarray_stats::QuantileExt; // Import QuantileExt for .min() and .max() on Array1

use log_data::LogRowData;
use constants::*;
use plotting_utils::*;
use step_response::*;

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file.csv>", args[0]);
        std::process::exit(1);
    }
    let input_file = &args[1];
    let input_path = Path::new(input_file);
    println!("Reading {}", input_file);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    let target_headers = [
        "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]",
        "axisI[0]", "axisI[1]", "axisI[2]",
        "axisD[0]", "axisD[1]", "axisD[2]",
        "setpoint[0]", "setpoint[1]", "setpoint[2]",
        "gyroADC[0]", "gyroADC[1]", "gyroADC[2]",
        "gyroUnfilt[0]", "gyroUnfilt[1]", "gyroUnfilt[2]",
        "debug[0]", "debug[1]", "debug[2]", "debug[3]",
        "throttle",
    ];

    // Flags to track if specific optional or plot-dependent headers are found.
    // These are used to decide which plots/calculations are possible.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[axis]" is present.
    let mut gyro_header_found = [false; 3]; // Tracks if "gyroADC[axis]" is present (filtered gyro).
    let mut gyro_unfilt_header_found = [false; 3]; // Tracks if "gyroUnfilt[axis]" is present.
    let mut debug_header_found = [false; 4]; // Tracks if "debug[idx]" is present.

    // Declare header_indices here so it's accessible outside the block
    let header_indices: Vec<Option<usize>>;

    // Read CSV header and map target headers to indices.
    { // Block to limit the scope of the file reader
        let file = File::open(input_file)?;
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
             let found = header_indices[i].is_some(); // Use header_indices here
             println!("  '{}': {}", name, if found { "Found" } else { "Not Found" });
             if !found { // Time (0), P (1-3), I (4-6), D[0-1] (7-8) are essential for PIDsum
                 essential_pid_headers_found = false;
             }
        }

        // Check optional 'axisD[2]' header (Index 9).
        let axis_d2_found_in_csv = header_indices[9].is_some(); // Use header_indices here
        println!("  '{}': {} (Optional, defaults to 0.0 if not found)", target_headers[9], if axis_d2_found_in_csv { "Found" } else { "Not Found" });

        // Check setpoint headers (Indices 10-12).
        for axis in 0..3 {
            setpoint_header_found[axis] = header_indices[10 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Setpoint vs PIDsum and Step Response Axis {})", target_headers[10 + axis], if setpoint_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check gyro (filtered) headers (Indices 13-15).
         for axis in 0..3 {
            gyro_header_found[axis] = header_indices[13 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Essential for Step Response and Gyro vs Unfilt Axis {})", target_headers[13 + axis], if gyro_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check gyroUnfilt headers (Indices 16-18).
        for axis in 0..3 {
            gyro_unfilt_header_found[axis] = header_indices[16 + axis].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for Gyro vs Unfilt Axis {})", target_headers[16 + axis], if gyro_unfilt_header_found[axis] { "Found" } else { "Not Found" }, axis);
        }

        // Check debug headers (Indices 19-22).
        for idx_offset in 0..4 {
            debug_header_found[idx_offset] = header_indices[19 + idx_offset].is_some(); // Use header_indices here
            println!("  '{}': {} (Fallback for gyroUnfilt[0-2])", target_headers[19 + idx_offset], if debug_header_found[idx_offset] { "Found" } else { "Not Found" });
        }

        // Check throttle header (Index 23).
        let throttle_found_in_csv = header_indices[23].is_some(); // Use header_indices here
        println!("  '{}': {} (Optional, for filtering)", target_headers[23], if throttle_found_in_csv { "Found" } else { "Not Found" });


        if !essential_pid_headers_found {
             let missing_essentials: Vec<String> = (0..=8).filter(|&i| header_indices[i].is_none()).map(|i| format!("'{}'", target_headers[i])).collect(); // Use header_indices here
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
    } // File reader is dropped here

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();
    println!("\nReading data rows...");
    { // Block to limit the scope of the file reader
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new().has_headers(true).trim(csv::Trim::All).from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    // Helper closure to parse a value from the record using the target header index.
                    // Uses the header_indices vector which is now in scope.
                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // Parse Time (us)
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    // Parse P, I, D, Setpoint, Gyro (filtered)
                    for axis in 0..3 {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // D term with optional axisD[2] fallback
                        let d_target_idx = 7 + axis;
                        // Use header_indices directly
                        if axis == 2 && header_indices[d_target_idx].is_none() {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(10 + axis);
                        }
                        if gyro_header_found[axis] {
                             current_row_data.gyro[axis] = parse_f64_by_target_idx(13 + axis);
                        }
                    }

                    // Parse gyroUnfilt and debug
                    let mut parsed_gyro_unfilt = [None; 3];
                    let mut parsed_debug = [None; 4];

                    for axis in 0..3 {
                        if gyro_unfilt_header_found[axis] {
                            parsed_gyro_unfilt[axis] = parse_f64_by_target_idx(16 + axis);
                        }
                    }

                    for idx_offset in 0..4 {
                        if debug_header_found[idx_offset] {
                            parsed_debug[idx_offset] = parse_f64_by_target_idx(19 + idx_offset);
                        }
                        current_row_data.debug[idx_offset] = parsed_debug[idx_offset];
                    }

                    // Apply Fallback Logic for gyro_unfilt
                    for axis in 0..3 {
                        current_row_data.gyro_unfilt[axis] = match parsed_gyro_unfilt[axis] {
                            Some(val) => Some(val),
                            None => match parsed_debug[axis] {
                                Some(val) => Some(val),
                                None => Some(0.0), // Default to 0.0 if both missing
                            }
                        };
                    }

                    // Parse Throttle
                    // Use header_indices directly
                    if header_indices[23].is_some() {
                        current_row_data.throttle = parse_f64_by_target_idx(23);
                    }

                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // File reader is dropped here


    println!("Finished reading {} data rows.", all_log_data.len());

    // --- Calculate Average Sample Rate ---
    let mut sample_rate: Option<f64> = None;
    let mut first_time: Option<f64> = None;
    let mut last_time: Option<f64> = None;

    if all_log_data.len() > 1 {
        let mut total_delta = 0.0;
        let mut count = 0;
        for i in 1..all_log_data.len() {
            if let (Some(t1), Some(t0)) = (all_log_data[i].time_sec, all_log_data[i-1].time_sec) {
                let delta = t1 - t0;
                if delta > 1e-9 {
                    total_delta += delta;
                    count += 1;
                }
            }
        }
        if count > 0 {
            let avg_delta = total_delta / count as f64;
            sample_rate = Some(1.0 / avg_delta);
            println!("Estimated Sample Rate: {:.2} Hz", sample_rate.unwrap());
        }

        first_time = all_log_data.first().and_then(|row| row.time_sec);
        last_time = all_log_data.last().and_then(|row| row.time_sec);
    }
    if sample_rate.is_none() {
         println!("Warning: Could not determine sample rate (need >= 2 data points with distinct timestamps). Step response calculation might fail.");
    }

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }

    // --- Data Preparation for Plots ---
    // This vector is specifically for the *first* PIDsum plot (which will now also show error and setpoint)
    // We collect all necessary data points here for that specific plot.
    let mut pidsum_setpoint_gyro_plot_data: [Vec<(f64, Option<f64>, Option<f64>, Option<f64>)>; 3] = [Vec::new(), Vec::new(), Vec::new()];

    // This vector is specifically for the *second* Setpoint vs PIDsum plot (keeps original logic)
    let mut setpoint_vs_pidsum_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_data_available_for_setpoint_vs_pidsum = [false; 3]; // Flag for the second plot

    // Data for other plots remains the same structure
    let mut gyro_vs_unfilt_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_vs_gyro_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut step_response_input_data: [(Vec<f64>, Vec<f32>, Vec<f32>); 3] = [
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
        (Vec::new(), Vec::new(), Vec::new()),
    ];

    let mut gyro_vs_unfilt_data_available = [false; 3];
    let mut setpoint_vs_gyro_data_available = [false; 3];
    let mut step_response_input_available = [false; 3];


    // Populate plot data structures.
    for row in &all_log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                let pidsum = row.p_term[axis_index].and_then(|p| {
                    row.i_term[axis_index].and_then(|i| {
                        row.d_term[axis_index].map(|d| p + i + d)
                    })
                });

                let setpoint = row.setpoint[axis_index];
                let gyro_filt = row.gyro[axis_index];
                let gyro_unfilt = row.gyro_unfilt[axis_index]; // Get unfiltered gyro

                // Data for the *first* PIDsum/Error/Setpoint plot
                if pidsum.is_some() || setpoint.is_some() || gyro_filt.is_some() {
                     // We push a point if *any* of the relevant values for this plot are available.
                     // The plot drawing logic will decide which lines to draw.
                    pidsum_setpoint_gyro_plot_data[axis_index].push((time, setpoint, gyro_filt, pidsum));
                }


                // Data for the *second* Setpoint vs PIDsum plot (original logic)
                if setpoint_header_found[axis_index] {
                     if let (Some(setpoint_val), Some(pidsum_val)) = (setpoint, pidsum) {
                        setpoint_vs_pidsum_data[axis_index].push((time, setpoint_val, pidsum_val));
                         setpoint_data_available_for_setpoint_vs_pidsum[axis_index] = true;
                     }
                 }

                 // Step Response Input Data (filtered by time and movement)
                 // Requires setpoint and filtered gyro
                 if setpoint.is_some() && gyro_filt.is_some() {
                     if let (Some(setpoint_val), Some(gyro_filt_val)) = (setpoint, gyro_filt) {
                         let mut include_point = false;
                         if let (Some(first), Some(last)) = (first_time, last_time) {
                             if time >= first + EXCLUDE_START_S && time <= last - EXCLUDE_END_S {
                                 if setpoint_val.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_filt_val.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                     include_point = true;
                                 }
                             }
                         } else { // Include if time range unknown, based on movement only
                             if setpoint_val.abs() >= MOVEMENT_THRESHOLD_DEG_S || gyro_filt_val.abs() >= MOVEMENT_THRESHOLD_DEG_S {
                                  include_point = true;
                             }
                         }

                         if include_point {
                             step_response_input_data[axis_index].0.push(time);
                             step_response_input_data[axis_index].1.push(setpoint_val as f32);
                             step_response_input_data[axis_index].2.push(gyro_filt_val as f32);
                             step_response_input_available[axis_index] = true;
                         }
                     }
                 }

                 // Gyro vs Unfiltered Gyro Data
                 // Requires filtered gyro and unfiltered gyro
                 if gyro_filt.is_some() && gyro_unfilt.is_some() {
                     if let (Some(gyro_filt_val), Some(gyro_unfilt_val)) = (gyro_filt, gyro_unfilt) {
                         gyro_vs_unfilt_data[axis_index].push((time, gyro_filt_val, gyro_unfilt_val));
                         gyro_vs_unfilt_data_available[axis_index] = true;
                     }
                 }

                 // Setpoint vs Gyro Data
                 // Requires setpoint and filtered gyro
                  if setpoint.is_some() && gyro_filt.is_some() {
                     if let (Some(setpoint_val), Some(gyro_filt_val)) = (setpoint, gyro_filt) {
                         setpoint_vs_gyro_data[axis_index].push((time, setpoint_val, gyro_filt_val));
                         setpoint_vs_gyro_data_available[axis_index] = true;
                     }
                 }
            }
        }
    }


    // --- Calculate Step Response Data ---
    println!("\n--- Calculating Step Response ---");
    // Store the raw QC'd stacked responses and setpoints for later averaging in main
    let mut step_response_calculation_results: [Option<(Array1<f64>, Array2<f32>, Array1<f32>)>; 3] = [None, None, None];


     if let Some(sr) = sample_rate {
        for axis_index in 0..3 {
            if step_response_input_available[axis_index] {
                println!("  Calculating step response for Axis {}...", axis_index);
                let time_arr = Array1::from(step_response_input_data[axis_index].0.clone());
                let setpoints_arr = Array1::from(step_response_input_data[axis_index].1.clone());
                let gyros_filtered_arr = Array1::from(step_response_input_data[axis_index].2.clone());

                let min_required_samples = (FRAME_LENGTH_S * sr).ceil() as usize;
                if time_arr.len() >= min_required_samples {
                    // Call the calculation function which now returns stacked QC'd responses
                    match calculate_step_response_python_style(&time_arr, &setpoints_arr, &gyros_filtered_arr, sr) {
                        Ok(result) => {
                             let num_qc_windows = result.1.shape()[0]; // Check number of QC windows
                             if num_qc_windows > 0 {
                                 step_response_calculation_results[axis_index] = Some(result); // Store the calculated data.
                                 println!("    ... Calculation successful for Axis {}. {} windows passed QC.", axis_index, num_qc_windows);
                             } else {
                                println!("    ... Calculation returned no valid windows for Axis {}. Skipping.", axis_index);
                             }
                        }
                        Err(e) => {
                            eprintln!("    ... Calculation failed for Axis {}: {}", axis_index, e);
                        }
                    }
                } else {
                     println!("    ... Skipping Axis {}: Not enough movement data points ({}) for windowing (need at least {}).", axis_index, time_arr.len(), min_required_samples);
                }
            } else {
                 println!("  Skipping Axis {}: Missing required input data (Setpoint or Filtered Gyro 'gyroADC').", axis_index);
            }
        }
    } else {
         println!("  Skipping Step Response Calculation: Sample rate could not be determined.");
    }


    // --- Generate Stacked PIDsum vs PID Error vs Setpoint Plot ---
    // This modifies the original PIDsum plot to add Setpoint and PID Error lines
    println!("\n--- Generating Stacked PIDsum vs PID Error vs Setpoint Plot ---");
    // Check if *any* axis has *any* data (PIDsum, Setpoint, or Gyro) collected for this plot
    if pidsum_setpoint_gyro_plot_data.iter().any(|v| !v.is_empty()) {
        let output_file_pidsum_error = format!("{}_PIDsum_PIDerror_Setpoint_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_pidsum_error = BitMapBackend::new(&output_file_pidsum_error, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_pidsum_error.fill(&WHITE)?;

        // Add main title on the full drawing area
        root_area_pidsum_error.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;

        // Create a margined area below the title for the subplots
        let margined_root_area_pidsum_error = root_area_pidsum_error.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_pidsum_error.split_evenly((3, 1));

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index];
            let axis_data = &pidsum_setpoint_gyro_plot_data[axis_index];

            // Collect actual data points for plotting this axis
            let mut pidsum_line_data: Vec<(f64, f64)> = Vec::new();
            let mut setpoint_line_data: Vec<(f64, f64)> = Vec::new();
            let mut pid_error_data: Vec<(f64, f64)> = Vec::new();

            for (time, setpoint, gyro_filt, pidsum) in axis_data {
                 if let Some(t) = time.into() { // time is guaranteed Some, but iter returns f64
                     if let Some(p) = pidsum {
                        pidsum_line_data.push((*t, *p));
                     }
                     if let Some(s) = setpoint {
                         setpoint_line_data.push((*t, *s));
                         if let Some(g) = gyro_filt {
                             pid_error_data.push((*t, s - g)); // Calculate PID Error
                         }
                     }
                 }
            }

            // Check if *any* data is available for this axis before drawing chart
            if !pidsum_line_data.is_empty() || !setpoint_line_data.is_empty() || !pid_error_data.is_empty() {

                 // Calculate combined range for all potential lines
                 let mut val_min = f64::INFINITY;
                 let mut val_max = f64::NEG_INFINITY;
                 let mut time_min = f64::INFINITY;
                 let mut time_max = f64::NEG_INFINITY;

                 for (t, v) in pidsum_line_data.iter() { val_min = val_min.min(*v); val_max = val_max.max(*v); time_min = time_min.min(*t); time_max = time_max.max(*t); }
                 for (t, v) in setpoint_line_data.iter() { val_min = val_min.min(*v); val_max = val_max.max(*v); time_min = time_min.min(*t); time_max = time_max.max(*t); }
                 for (t, v) in pid_error_data.iter() { val_min = val_min.min(*v); val_max = val_max.max(*v); time_min = time_min.min(*t); time_max = time_max.max(*t); }

                 if time_min.is_infinite() || val_min.is_infinite() {
                      // This case should ideally not be reached if the outer check passed, but good defensive coding
                      draw_unavailable_message(area, axis_index, "PIDsum/PIDerror/Setpoint")?;
                      continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} PIDsum vs PID Error vs Setpoint", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let mut series_drawn = 0;

                // Draw PIDsum line if data available
                if !pidsum_line_data.is_empty() {
                     let pidsum_color = Palette99::pick(COLOR_PIDSUM);
                     chart.draw_series(LineSeries::new(
                         pidsum_line_data.iter().cloned(),
                         &pidsum_color,
                     ))?
                     .label("PIDsum (P+I+D)")
                     .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pidsum_color.stroke_width(2)));
                     series_drawn += 1;
                }

                // Draw PID Error line if data available
                if !pid_error_data.is_empty() {
                    let error_color = COLOR_PID_ERROR; // Use the new color constant
                    chart.draw_series(LineSeries::new(
                        pid_error_data.iter().cloned(),
                        error_color.stroke_width(1), // Use stroke_width for visibility
                    ))?
                    .label("PID Error (Setpoint - GyroADC)")
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], error_color.stroke_width(2)));
                    series_drawn += 1;
                }

                // Draw Setpoint line if data available
                if !setpoint_line_data.is_empty() {
                    let setpoint_color = Palette99::pick(COLOR_SETPOINT); // Use existing Setpoint color constant
                    chart.draw_series(LineSeries::new(
                        setpoint_line_data.iter().cloned(),
                        &setpoint_color,
                    ))?
                    .label("Setpoint")
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], setpoint_color.stroke_width(2)));
                    series_drawn += 1;
                }

                // Configure and draw the legend only if at least one series was drawn
                if series_drawn > 0 {
                    chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                         .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
                }

            } else {
                // No data available for any of the lines for this axis
                println!("  INFO: No data (PIDsum, Setpoint, or Gyro) available for Axis {} for combined plot. Drawing placeholder.", axis_index);
                draw_unavailable_message(area, axis_index, "PIDsum/PIDerror/Setpoint")?;
            }
        }
        root_area_pidsum_error.present()?;
        println!("  Stacked PIDsum vs PID Error vs Setpoint plot saved as '{}'.", output_file_pidsum_error);
    } else {
        println!("  Skipping Stacked PIDsum vs PID Error vs Setpoint Plot: No relevant data available for any axis.");
    }


    // --- Generate Stacked Setpoint vs PIDsum Plot (Original Second Plot) ---
    println!("\n--- Generating Stacked Setpoint vs PIDsum Plot ---");
    if setpoint_data_available_for_setpoint_vs_pidsum.iter().any(|&x| x) {
        let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_setpoint = BitMapBackend::new(&output_file_setpoint, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_setpoint.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_setpoint = root_area_setpoint.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_setpoint.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if setpoint_data_available_for_setpoint_vs_pidsum[axis_index] && !setpoint_vs_pidsum_data[axis_index].is_empty() {
                let (time_min, time_max) = setpoint_vs_pidsum_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_pidsum_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| (min_y.min(*s).min(*p), max_y.max(*s).max(*p)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs PIDsum", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let sp_color = Palette99::pick(COLOR_SETPOINT);
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, s, _p)| (*t, *s)),
                    &sp_color,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color.stroke_width(2)));

                let pid_color = Palette99::pick(COLOR_PIDSUM_VS_SETPOINT);
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, _s, p)| (*t, *p)),
                    &pid_color,
                ))?
                .label("PIDsum")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_color.stroke_width(2)));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                println!("  INFO: No Setpoint vs PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                 draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
            }
        }
        root_area_setpoint.present()?;
        println!("  Stacked Setpoint vs PIDsum plot saved as '{}'.", output_file_setpoint);
    } else {
        println!("  Skipping Stacked Setpoint vs PIDsum Plot: No Setpoint vs PIDsum data available for any axis.");
    }

    // --- Generate Stacked Setpoint vs Gyro Plot ---
    println!("\n--- Generating Stacked Setpoint vs Gyro Plot ---");
    if setpoint_vs_gyro_data_available.iter().any(|&x| x) {
        let output_file_setpoint_gyro = format!("{}_SetpointVsGyro_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_setpoint_gyro = BitMapBackend::new(&output_file_setpoint_gyro, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint_gyro.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_setpoint_gyro.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_setpoint_gyro = root_area_setpoint_gyro.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_setpoint_gyro.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if setpoint_vs_gyro_data_available[axis_index] && !setpoint_vs_gyro_data[axis_index].is_empty() {
                let (time_min, time_max) = setpoint_vs_gyro_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_gyro_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, g)| (min_y.min(*s).min(*g), max_y.max(*s).max(*g)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/Gyro")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs Gyro", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                    let gyro_color = Palette99::pick(COLOR_STEP_RESPONSE_LOW_SP); // Use the same blue/green as low setpoint step response
                    chart.draw_series(LineSeries::new(
                        setpoint_vs_gyro_data[axis_index].iter().map(|(t, _s, g)| (*t, *g)),
                        &gyro_color,
                    ))?
                    .label("Gyro (gyroADC)")
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], gyro_color.stroke_width(2)));

                    let sp_color = COLOR_STEP_RESPONSE_HIGH_SP; // Use the same orange as high setpoint step response
                chart.draw_series(LineSeries::new(
                    setpoint_vs_gyro_data[axis_index].iter().map(|(t, s, _g)| (*t, *s)),
                    &sp_color,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color.stroke_width(2)));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                println!("  INFO: No Setpoint vs Gyro data available for Axis {}. Drawing placeholder.", axis_index);
                 draw_unavailable_message(area, axis_index, "Setpoint/Gyro")?;
            }
        }
        root_area_setpoint_gyro.present()?;
        println!("  Stacked Setpoint vs Gyro plot saved as '{}'.", output_file_setpoint_gyro);
    } else {
        println!("  Skipping Stacked Setpoint vs Gyro Plot: No Setpoint vs Gyro data available for any axis.");
    }

    // --- Generate Stacked Step Response Plot ---
    println!("\n--- Generating Stacked Step Response Plot ---");
    // Check if *any* axis has step response calculation results stored
    if step_response_calculation_results.iter().any(|x| x.is_some()) {
        // Create the main drawing area for the entire plot
        let output_file_step = format!("{}_step_response_stacked_plot_{}s.png", root_name, STEP_RESPONSE_PLOT_DURATION_S);
        let root_area_step = BitMapBackend::new(&output_file_step, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_step.fill(&WHITE)?;

        // Add main title on the full drawing area
        root_area_step.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_step = root_area_step.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_step.split_evenly((3, 1));

        // Get sample rate for steady-state window calculation (fallback if needed)
        let sr = sample_rate.unwrap_or(1000.0); // Use a reasonable default if sample rate unknown
        let ss_start_sample = (STEADY_STATE_START_S * sr).floor() as usize;
        let ss_end_sample = (STEADY_STATE_END_S * sr).ceil() as usize;

        // Tolerance for checking if steady-state is near 1.0
        const STEADY_STATE_TOLERANCE: f64 = 0.2; // Allow steady state between 0.8 and 1.2

        for axis_index in 0..3 {
            // Retrieve the raw QC'd stacked responses and setpoints
            if let Some((response_time, valid_stacked_responses, valid_window_max_setpoints)) = &step_response_calculation_results[axis_index] {
                let area = &sub_plot_areas[axis_index];
                let response_length_samples = response_time.len(); // Get actual length from the time array

                // Ensure response_time is not empty before proceeding
                if response_length_samples == 0 {
                     draw_unavailable_message(area, axis_index, "Step Response (Empty Time Data)")?;
                     continue;
                 }

                let num_qc_windows = valid_stacked_responses.shape()[0];
                 if num_qc_windows == 0 {
                     draw_unavailable_message(area, axis_index, "Step Response (No Valid Windows)")?;
                     continue;
                 }


                // 3. Implement Setpoint Masking for QC Windows.
                let low_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v < SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });
                let high_mask: Array1<f32> = valid_window_max_setpoints.mapv(|v| if v >= SETPOINT_THRESHOLD as f32 { 1.0 } else { 0.0 });
                let combined_mask: Array1<f32> = Array1::ones(num_qc_windows); // Mask for all QC windows


                // 4. Implement Averaging (using mean).
                let mut low_response_avg = Array1::<f64>::zeros(0);
                let mut high_response_avg = Array1::<f64>::zeros(0);
                let mut combined_response_avg = Array1::<f64>::zeros(0);

                // Always attempt low and combined if QC windows exist
                if low_mask.iter().any(|&w| w > 0.0) {
                    match average_responses( // CAverage_responses
                        &valid_stacked_responses,
                        &low_mask,
                        response_length_samples,
                    ) {
                        Ok(resp) => low_response_avg = resp,
                        Err(e) => eprintln!("    ... Low setpoint averaging failed for Axis {}: {}", axis_index, e),
                    }
                } else {
                     println!("  INFO: No low setpoint windows for Axis {}. Skipping low response averaging.", axis_index);
                }

                // Only attempt high if high setpoint windows exist
                if high_mask.iter().any(|&w| w > 0.0) {
                    match average_responses( // Average_responses
                        &valid_stacked_responses,
                        &high_mask,
                        response_length_samples,
                    ) {
                        Ok(resp) => high_response_avg = resp,
                        Err(e) => eprintln!("    ... High setpoint averaging failed for Axis {}: {}", axis_index, e),
                    }
                } else {
                     println!("  INFO: No high setpoint windows for Axis {}. Skipping high response averaging.", axis_index);
                }

                // Combined response should always be attempted if there are QC windows
                match average_responses( // Average_responses
                   &valid_stacked_responses,
                   &combined_mask,
                   response_length_samples,
                ) {
                   Ok(resp) => combined_response_avg = resp,
                   Err(e) => eprintln!("    ... Combined averaging failed for Axis {}: {}", axis_index, e),
                }


                // Apply smoothing *before* shifting and final normalization
                let smoothed_low_response = moving_average_smooth_f64(&low_response_avg, POST_AVERAGING_SMOOTHING_WINDOW);
                let smoothed_high_response = moving_average_smooth_f64(&high_response_avg, POST_AVERAGING_SMOOTHING_WINDOW);
                let smoothed_combined_response = moving_average_smooth_f64(&combined_response_avg, POST_AVERAGING_SMOOTHING_WINDOW);


                // --- Shift the smoothed curves to start at 0 and Normalize to settle at 1.0 ---
                let current_ss_start_sample = ss_start_sample.min(response_length_samples);
                let current_ss_end_sample = ss_end_sample.min(response_length_samples);

                let mut final_low_response = Array1::<f64>::zeros(0); // Start with empty/invalid
                let mut is_low_response_valid = false;
                if !smoothed_low_response.is_empty() && current_ss_start_sample < current_ss_end_sample {
                    let mut shifted_response = smoothed_low_response.clone();
                    let first_val = shifted_response[0];
                    shifted_response.mapv_inplace(|v| v - first_val); // Shift to start at 0

                    let steady_state_segment = shifted_response.slice(s![current_ss_start_sample..current_ss_end_sample]);
                    if let Some(steady_state_mean) = steady_state_segment.mean() {
                        if steady_state_mean.abs() > 1e-9 { // Avoid division by zero for normalization
                            let normalized_response = shifted_response.mapv(|v| v / steady_state_mean);
                            // Check if the steady state of the *normalized* response is near 1.0
                            if let Some(normalized_ss_mean) = normalized_response.slice(s![current_ss_start_sample..current_ss_end_sample]).mean() {
                                 if (normalized_ss_mean - 1.0).abs() <= STEADY_STATE_TOLERANCE {
                                     final_low_response = normalized_response;
                                     is_low_response_valid = true;
                                 } else {
                                     println!("  INFO: Axis {} low response steady-state ({:.2}) outside tolerance after normalization. Skipping plot.", axis_index, normalized_ss_mean);
                                 }
                            } else {
                                 eprintln!("Warning: Could not calculate normalized steady-state mean for Axis {} low response. Skipping plot.", axis_index);
                             }
                         } else {
                             println!("  INFO: Axis {} low response steady-state mean near zero after shifting. Skipping final normalization and plot.", axis_index);
                         }
                     } else {
                          eprintln!("Warning: Could not calculate steady-state mean for Axis {} low response after shifting. Skipping final normalization and plot.", axis_index);
                     }
                 } else {
                      println!("  INFO: Axis {} low response data empty or steady-state window invalid after smoothing/shifting. Skipping plot.", axis_index);
                 }


                let mut final_high_response = Array1::<f64>::zeros(0); // Start with empty/invalid
                let mut is_high_response_valid = false;
                if !smoothed_high_response.is_empty() && current_ss_start_sample < current_ss_end_sample {
                     let mut shifted_response = smoothed_high_response.clone();
                     let first_val = shifted_response[0];
                     shifted_response.mapv_inplace(|v| v - first_val); // Shift to start at 0

                     let steady_state_segment = shifted_response.slice(s![current_ss_start_sample..current_ss_end_sample]);
                     if let Some(steady_state_mean) = steady_state_segment.mean() {
                         if steady_state_mean.abs() > 1e-9 {
                             let normalized_response = shifted_response.mapv(|v| v / steady_state_mean);
                             if let Some(normalized_ss_mean) = normalized_response.slice(s![current_ss_start_sample..current_ss_end_sample]).mean() {
                                  if (normalized_ss_mean - 1.0).abs() <= STEADY_STATE_TOLERANCE {
                                      final_high_response = normalized_response;
                                      is_high_response_valid = true;
                                  } else {
                                      println!("  INFO: Axis {} high response steady-state ({:.2}) outside tolerance after normalization. Skipping plot.", axis_index, normalized_ss_mean);
                                  }
                             } else {
                                  eprintln!("Warning: Could not calculate normalized steady-state mean for Axis {} high response. Skipping plot.", axis_index);
                             }
                         } else {
                             println!("  INFO: Axis {} high response steady-state mean near zero after shifting. Skipping final normalization and plot.", axis_index);
                         }
                     } else {
                          eprintln!("Warning: Could not calculate steady-state mean for Axis {} high response after shifting. Skipping final normalization and plot.", axis_index);
                     }
                } else {
                     println!("  INFO: Axis {} high response data empty or steady-state window invalid after smoothing/shifting. Skipping plot.", axis_index);
                }


                let mut final_combined_response = Array1::<f64>::zeros(0); // Start with empty/invalid
                let mut is_combined_response_valid = false;
                if !smoothed_combined_response.is_empty() && current_ss_start_sample < current_ss_end_sample {
                     let mut shifted_response = smoothed_combined_response.clone();
                     let first_val = shifted_response[0];
                     shifted_response.mapv_inplace(|v| v - first_val); // Shift to start at 0

                     let steady_state_segment = shifted_response.slice(s![current_ss_start_sample..current_ss_end_sample]);
                     if let Some(steady_state_mean) = steady_state_segment.mean() {
                         if steady_state_mean.abs() > 1e-9 {
                             let normalized_response = shifted_response.mapv(|v| v / steady_state_mean);
                             if let Some(normalized_ss_mean) = normalized_response.slice(s![current_ss_start_sample..current_ss_end_sample]).mean() {
                                  if (normalized_ss_mean - 1.0).abs() <= STEADY_STATE_TOLERANCE {
                                      final_combined_response = normalized_response;
                                      is_combined_response_valid = true;
                                  } else {
                                      println!("  INFO: Axis {} combined response steady-state ({:.2}) outside tolerance after normalization. Skipping plot.", axis_index, normalized_ss_mean);
                                  }
                             } else {
                                  eprintln!("Warning: Could not calculate normalized steady-state mean for Axis {} combined response. Skipping plot.", axis_index);
                             }
                         } else {
                             println!("  INFO: Axis {} combined response steady-state mean near zero after shifting. Skipping final normalization and plot.", axis_index);
                         }
                     } else {
                          eprintln!("Warning: Could not calculate steady-state mean for Axis {} combined response after shifting. Skipping final normalization and plot.", axis_index);
                     }
                } else {
                     println!("  INFO: Axis {} combined response data empty or steady-state window invalid after smoothing/shifting. Skipping plot.", axis_index);
                }
                // --- End of normalization and validity check section ---


                // Determine plot range based *only* on valid responses
                let mut resp_min = f64::INFINITY;
                let mut resp_max = f64::NEG_INFINITY;

                if is_low_response_valid {
                    if let Ok(min_val) = final_low_response.min() { resp_min = resp_min.min(*min_val); }
                    if let Ok(max_val) = final_low_response.max() { resp_max = resp_max.max(*max_val); }
                }
                if is_high_response_valid {
                    if let Ok(min_val) = final_high_response.min() { resp_min = resp_min.min(*min_val); }
                    if let Ok(max_val) = final_high_response.max() { resp_max = resp_max.max(*max_val); }
                }
                if is_combined_response_valid {
                    if let Ok(min_val) = final_combined_response.min() { resp_min = resp_min.min(*min_val); }
                    if let Ok(max_val) = final_combined_response.max() { resp_max = resp_max.max(*max_val); }
                }

                // If no responses are valid, draw unavailable message
                if resp_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Step Response (No Valid Data)")?;
                     continue; // Skip drawing chart for this axis
                }

                let (final_resp_min, final_resp_max) = calculate_range(resp_min, resp_max);
                let final_time_max = STEP_RESPONSE_PLOT_DURATION_S * 1.05;

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Step Response (~{}s)", axis_index, STEP_RESPONSE_PLOT_DURATION_S), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(0f64..final_time_max.max(1e-9), final_resp_min..final_resp_max)?;

                chart.configure_mesh()
                    .x_desc("Time (s) relative to response start")
                    .y_desc("Normalized Response")
                    .x_labels(8)
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                // Draw high setpoint response (only if valid)
                if is_high_response_valid {
                    let high_sp_color = COLOR_STEP_RESPONSE_HIGH_SP;
                    chart.draw_series(LineSeries::new(
                        response_time.iter().zip(final_high_response.iter()).map(|(&t, &v)| (t, v)),
                        high_sp_color.stroke_width(2),
                    ))?
                    .label(format!("\u{2265} {} deg/s", SETPOINT_THRESHOLD))
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], high_sp_color.stroke_width(2)));
                }

                // Draw combined response (only if valid)
                if is_combined_response_valid {
                    let combined_color = COLOR_STEP_RESPONSE_COMBINED;
                    chart.draw_series(LineSeries::new(
                        response_time.iter().zip(final_combined_response.iter()).map(|(&t, &v)| (t, v)),
                        combined_color.stroke_width(2),
                    ))?
                    .label("Combined")
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], combined_color.stroke_width(2)));
                }

                // Draw low setpoint response (only if valid)
                if is_low_response_valid {
                    let low_sp_color = Palette99::pick(COLOR_STEP_RESPONSE_LOW_SP);
                    chart.draw_series(LineSeries::new(
                        response_time.iter().zip(final_low_response.iter()).map(|(&t, &v)| (t, v)),
                        low_sp_color.stroke_width(2),
                    ))?
                    .label(format!("< {} deg/s", SETPOINT_THRESHOLD))
                    .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], low_sp_color.stroke_width(2)));
                }

                // Configure and draw the legend.
                // Only draw legend if at least one series was drawn
                if is_low_response_valid || is_high_response_valid || is_combined_response_valid {
                     chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                         .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
                }


            } else {
                // Declare area here to be in scope for the unavailable message
                let area = &sub_plot_areas[axis_index];
                let reason = if !setpoint_header_found[axis_index] || !gyro_header_found[axis_index] {
                    "Setpoint/gyroADC Header Missing"
                 } else if sample_rate.is_none() {
                    "Sample Rate Unknown"
                 } else if !step_response_input_available[axis_index] {
                     "Input Data Missing/Invalid"
                 } else {
                     "Calculation Failed/No Data"
                 };
                println!("  INFO: No Step Response data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Step Response ({})", reason))?;
            }
        }
        root_area_step.present()?;
        println!("  Stacked Step Response plot saved as '{}'. (Duration: {}s)", output_file_step, STEP_RESPONSE_PLOT_DURATION_S);

    } else {
        println!("  Skipping Stacked Step Response Plot: No step response data could be calculated for any axis.");
    }

    // --- Generate Stacked Gyro vs Unfiltered Gyro Plot ---
    println!("\n--- Generating Stacked Gyro vs Unfiltered Gyro Plot ---");
    if gyro_vs_unfilt_data_available.iter().any(|&x| x) {
        let output_file_gyro = format!("{}_GyroVsUnfilt_stacked.png", root_name);
        // Create the main drawing area for the entire plot
        let root_area_gyro = BitMapBackend::new(&output_file_gyro, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_gyro.fill(&WHITE)?;

         // Add main title on the full drawing area
         root_area_gyro.draw(&Text::new(
            root_name.as_ref(),
            (10, 10), // Position near top-left
            ("sans-serif", 24).into_font().color(&BLACK),
        ))?;
        // Create a margined area below the title for the subplots
        let margined_root_area_gyro = root_area_gyro.margin(50, 5, 5, 5); // Top margin 50px

        // Split the margined area into subplots
        let sub_plot_areas = margined_root_area_gyro.split_evenly((3, 1));

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index];
            if gyro_vs_unfilt_data_available[axis_index] && !gyro_vs_unfilt_data[axis_index].is_empty() {
                let (time_min, time_max) = gyro_vs_unfilt_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = gyro_vs_unfilt_data[axis_index].iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, gf, gu)| (min_y.min(*gf).min(*gu), max_y.max(*gf).max(*gu)) );

                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Gyro/UnfiltGyro")?;
                     continue;
                 }

                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Filtered vs Unfiltered Gyro", axis_index), ("sans-serif", 20))
                    .margin(5).x_label_area_size(30).y_label_area_size(50)
                    .build_cartesian_2d(time_min..time_max, final_value_min..final_value_max)?;
                chart.configure_mesh().x_desc("Time (s)").y_desc("Gyro Value").x_labels(10).y_labels(5)
                    .light_line_style(&WHITE.mix(0.7)).label_style(("sans-serif", 12)).draw()?;

                let unfilt_color = Palette99::pick(COLOR_GYRO_UNFILT);
                chart.draw_series(LineSeries::new(
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, _gf, gu)| (*t, *gu)),
                    &unfilt_color,
                ))?
                .label("Unfiltered Gyro (gyroUnfilt/debug)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], unfilt_color.stroke_width(2)));

                let filt_color = Palette99::pick(COLOR_GYRO_FILT).filled();
                chart.draw_series(LineSeries::new(
                    gyro_vs_unfilt_data[axis_index].iter().map(|(t, gf, _gu)| (*t, *gf)),
                    filt_color,
                ))?
                .label("Filtered Gyro (gyroADC)")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], filt_color.stroke_width(2)));

                chart.configure_series_labels().position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8)).border_style(&BLACK).label_font(("sans-serif", 12)).draw()?;
            } else {
                let reason = if !gyro_header_found[axis_index] {
                    "gyroADC Header Missing"
                 } else {
                    "No Valid Data Rows"
                 };
                println!("  INFO: No Gyro vs Unfiltered Gyro data available for Axis {}: {}. Drawing placeholder.", axis_index, reason);
                 draw_unavailable_message(area, axis_index, &format!("Gyro/UnfiltGyro ({})", reason))?;
            }
        }
        root_area_gyro.present()?;
        println!("  Stacked Gyro vs Unfiltered Gyro plot saved as '{}'.", output_file_gyro);
    } else {
        println!("  Skipping Stacked Gyro vs Unfiltered Gyro Plot: No data available for any axis.");
    }

    println!();
    Ok(())
}