use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

// Structure to hold the data needed for calculating PID output per row
#[derive(Debug, Default, Clone)]
struct LogRowData {
    time_sec: Option<f64>,
    p_term: [Option<f64>; 3], // Assumes axisP[x] is the P *term*
    i_term: [Option<f64>; 3], // Assumes axisI[x] is the I *term*
    d_term: [Option<f64>; 3], // Assumes axisD[x] is the D *term*
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file.csv>", args[0]);
        std::process::exit(1);
    }
    let input_file = &args[1];
    let input_path = Path::new(input_file);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    // Headers needed: time and the P, I, D *terms* (components of the output)
    let target_headers = [
        // Time (Essential for X-axis)
        "time (us)",    // 0
        // P Term Components (Essential)
        "axisP[0]",     // 1
        "axisP[1]",     // 2
        "axisP[2]",     // 3
        // I Term Components (Essential)
        "axisI[0]",     // 4
        "axisI[1]",     // 5
        "axisI[2]",     // 6
        // D Term Components (Essential, except AxisD[2])
        "axisD[0]",     // 7
        "axisD[1]",     // 8
        "axisD[2]",     // 9 - Optional, defaults to 0 if missing
    ];

    // Flag to track if AxisD[2] header exists
    let mut axis_d2_header_found = false;

    let header_indices = { // Scope for header reading
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        let indices: Vec<Option<usize>> = target_headers
            .iter()
            .map(|&target_header| {
                header_record.iter().position(|h| h == target_header)
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut essential_headers_found = true;

        // Check essential headers (indices 0 through 8)
        for i in 0..=8 { // Check up to axisD[1]
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_headers_found = false;
                    "Not Found (Essential!)".to_string()
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }

        // Check optional header AxisD[2] (index 9) separately
        let axis_d2_name = target_headers[9];
        let axis_d2_status = match indices[9] {
            Some(idx) => {
                axis_d2_header_found = true; // Mark as found
                format!("Found at index {}", idx)
            }
            None => {
                // Not found, but this is okay. We'll default to 0.
                "Not Found (Optional, will default to 0.0)".to_string()
            }
        };
        println!("  '{}' (Target Index {}): {}", axis_d2_name, 9, axis_d2_status);


        if !essential_headers_found {
             // Construct a more specific error message
             let missing_essentials: Vec<String> = (0..=8)
                 .filter(|&i| indices[i].is_none())
                 .map(|i| format!("'{}'", target_headers[i]))
                 .collect();
             return Err(format!("Error: Missing essential headers: {}. Aborting.", missing_essentials.join(", ")).into());
        }
        indices // Return indices
    };

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();

    println!("\nReading P/I/D term data from CSV...");
    { // Scope for the reader
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    // Helper to parse f64 field safely using the resolved header index
                    let mut parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx) // Get Option<Option<usize>>
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref()) // Get Option<&usize>
                            .and_then(|&csv_idx| record.get(csv_idx)) // Get Option<&str>
                            .and_then(|val_str| val_str.parse::<f64>().ok()) // Get Option<f64>
                    };


                    // --- Parse Time ---
                    let time_us = parse_f64_by_target_idx(0); // Target index 0 = "time (us)"
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         // Time is essential, skip row if missing or invalid
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    // --- Parse P, I, D Terms for each axis ---
                    for axis in 0..3 {
                        // P term indices: 1, 2, 3 (Essential)
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        // I term indices: 4, 5, 6 (Essential)
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // D term indices: 7, 8, 9 (Handle axis 2 specially)
                        let d_target_idx = 7 + axis;
                        if axis == 2 && !axis_d2_header_found {
                             // Special case: AxisD[2] header was missing, default to 0.0
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             // Normal case: Parse using the found index (if header exists)
                             // or handle missing essential D[0]/D[1] header (will result in None)
                             // or handle parse error for D[2] even if header exists (will result in None)
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }
                    }

                     // Add the fully parsed row data to our main vector
                     // Note: If essential P/I/D[0]/D[1] terms were missing *in this row*
                     // (even if headers exist), they will be None here. The plotting
                     // filter_map will handle skipping such rows for the affected axis.
                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // Reader goes out of scope, file is closed

    println!("Finished reading {} data rows.", all_log_data.len());
    if !axis_d2_header_found {
        println!("INFO: 'axisD[2]' header was not found in the CSV. Used 0.0 as default value for Axis 2 D-term calculation.");
    }


    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }


    // --- Plotting Calculated PID Output ---
    println!("\n--- Generating Calculated PID Output Plots (P+I+D vs Time) ---");

    for axis_index in 0..3 {
        println!("Processing Axis {}...", axis_index);

        // Calculate PID output (P+I+D) for each time step where all components are available
        // This logic remains the same. If D[2] was defaulted to Some(0.0), it will be included.
        // If any *other* P/I/D term was None (due to missing header or parse error),
        // the row will be skipped for that axis's plot by filter_map.
        let pid_output_data: Vec<(f64, f64)> = all_log_data.iter().filter_map(|row| {
            // Ensure we have time and all three terms for this axis
            if let (Some(time), Some(p), Some(i), Some(d)) =
                (row.time_sec, row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
            {
                Some((time, p + i + d)) // Calculate total PID output
            } else {
                None // Skip row if any essential component for this axis is missing for this row
            }
        }).collect();


        if pid_output_data.is_empty() {
            println!("Skipping Axis {}: No rows found with complete P, I, and D term data.", axis_index);
            continue;
        }

        // --- Determine Plot Ranges ---
         let (time_min, time_max) = pid_output_data.iter()
             .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));

         let (output_min, output_max) = pid_output_data.iter()
              .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));


        // Add padding
        let y_range = (output_max - output_min).abs();
        let y_padding = if y_range < 1e-6 { 0.5 } else { y_range * 0.1 };
        let final_y_min = output_min - y_padding;
        let final_y_max = output_max + y_padding;


        // --- Setup Plot ---
        let output_file = format!("{}_axis{}_calculated_pid_output.png", root_name, axis_index);
        let root_area = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let title = format!("Axis {} Calculated PID Output (Sum of P, I, D Terms from Log)", axis_index);

        let mut chart = ChartBuilder::on(&root_area)
            .caption(title, ("sans-serif", 30))
            .margin(15)
            .x_label_area_size(50)
            .y_label_area_size(70)
            .build_cartesian_2d(
                time_min..time_max,
                final_y_min..final_y_max,
            )?;

        chart.configure_mesh()
            .x_desc("Time (s)")
            .y_desc(format!("Axis {} Calculated PID Output", axis_index))
            .x_labels(10)
            .y_labels(10)
            .light_line_style(&WHITE.mix(0.7))
            .draw()?;

        // --- Draw Series ---

        // Draw Calculated PID Output Line (Green)
        chart.draw_series(LineSeries::new(
            pid_output_data, // Use the calculated (time, P+I+D) vector
            &GREEN,
        ))?
        .label("PID Output (P+I+D)")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

        chart.configure_series_labels()
            .position(SeriesLabelPosition::UpperRight)
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        println!("Axis {} calculated PID output plot saved as '{}'.", axis_index, output_file);

    } // End axis plotting loop

    Ok(())
}
