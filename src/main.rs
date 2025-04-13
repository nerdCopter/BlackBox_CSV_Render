use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

// Structure to hold the data needed for calculating PIDsum and setpoint comparison per row
#[derive(Debug, Default, Clone)]
struct LogRowData {
    time_sec: Option<f64>,
    p_term: [Option<f64>; 3],   // Assumes axisP[x] is the P *term*
    i_term: [Option<f64>; 3],   // Assumes axisI[x] is the I *term*
    d_term: [Option<f64>; 3],   // Assumes axisD[x] is the D *term*
    setpoint: [Option<f64>; 3], // Assumes setpoint[x] is the target value
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
    // Headers needed: time, P/I/D terms, and setpoints
    let target_headers = [
        // Time (Essential for X-axis)
        "time (us)",    // 0
        // P Term Components (Essential for PIDsum plot)
        "axisP[0]",     // 1
        "axisP[1]",     // 2
        "axisP[2]",     // 3
        // I Term Components (Essential for PIDsum plot)
        "axisI[0]",     // 4
        "axisI[1]",     // 5
        "axisI[2]",     // 6
        // D Term Components (Essential for PIDsum plot, except AxisD[2])
        "axisD[0]",     // 7
        "axisD[1]",     // 8
        "axisD[2]",     // 9 - Optional, defaults to 0 if missing for PIDsum plot
        // Setpoint Components (Essential for Setpoint vs PIDsum plot)
        "setpoint[0]",  // 10
        "setpoint[1]",  // 11
        "setpoint[2]",  // 12
    ];

    // Flags to track if specific headers exist
    let mut axis_d2_header_found = false;
    let mut setpoint_header_found = [false; 3]; // Track presence for each axis

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
        let mut essential_pid_headers_found = true;

        // Check essential headers for PIDsum plot (indices 0 through 8)
        for i in 0..=8 { // Check up to axisD[1]
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_pid_headers_found = false;
                    format!("Not Found (Essential for PIDsum Plot!)")
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
                format!("Not Found (Optional for PIDsum plot, will default to 0.0)")
            }
        };
        println!("  '{}' (Target Index {}): {}", axis_d2_name, 9, axis_d2_status);

        // Check setpoint headers (indices 10, 11, 12) - essential for Setpoint vs PIDsum plot
        for axis in 0..3 {
            let target_idx = 10 + axis;
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    setpoint_header_found[axis] = true; // Mark as found for this axis
                    format!("Found at index {}", idx)
                 }
                 None => {
                    format!("Not Found (Essential for Setpoint vs PIDsum Plot Axis {})", axis) // Changed message
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }


        if !essential_pid_headers_found {
             // Construct a more specific error message for PIDsum plot headers
             let missing_essentials: Vec<String> = (0..=8)
                 .filter(|&i| indices[i].is_none())
                 .map(|i| format!("'{}'", target_headers[i]))
                 .collect();
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
        indices // Return indices
    };

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();

    println!("\nReading P/I/D term and Setpoint data from CSV...");
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

                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<f64>().ok())
                    };

                    // --- Parse Time ---
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue;
                    }

                    // --- Parse P, I, D Terms and Setpoint for each axis ---
                    for axis in 0..3 {
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        let d_target_idx = 7 + axis;
                        if axis == 2 && !axis_d2_header_found {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        let setpoint_target_idx = 10 + axis;
                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(setpoint_target_idx);
                        }
                    }

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
        println!("INFO: 'axisD[2]' header was not found. Used 0.0 as default value for Axis 2 D-term calculation in PIDsum plot.");
    }
    for axis in 0..3 {
        if !setpoint_header_found[axis] {
             println!("INFO: 'setpoint[{}]' header was not found. Setpoint vs PIDsum plot for Axis {} cannot be generated.", axis, axis); // Changed message
        }
    }


    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }


    // --- Plotting Loop for Each Axis ---
    println!("\n--- Generating Plots for Each Axis ---");

    for axis_index in 0..3 {
        println!("Processing Axis {}...", axis_index);

        // --- 1. Calculate and Plot PIDsum (P+I+D vs Time) ---
        { // Scope for PIDsum Plot
            println!("  Generating PIDsum plot (P+I+D vs Time)...");
            let pid_output_data: Vec<(f64, f64)> = all_log_data.iter().filter_map(|row| {
                if let (Some(time), Some(p), Some(i), Some(d)) =
                    (row.time_sec, row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    Some((time, p + i + d))
                } else {
                    None
                }
            }).collect();

            if pid_output_data.is_empty() {
                println!("  Skipping Axis {} PIDsum Plot: No rows found with complete P, I, and D term data.", axis_index);
            } else {
                let (time_min, time_max) = pid_output_data.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
                let (output_min, output_max) = pid_output_data.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));
                let y_range = (output_max - output_min).abs();
                let y_padding = if y_range < 1e-6 { 0.5 } else { y_range * 0.1 };
                let final_y_min = output_min - y_padding;
                let final_y_max = output_max + y_padding;

                let output_file = format!("{}_axis{}_PIDsum.png", root_name, axis_index);
                let root_area = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
                root_area.fill(&WHITE)?;

                let title = format!("Axis {} PIDsum (P+I+D)", axis_index);
                let mut chart = ChartBuilder::on(&root_area)
                    .caption(title, ("sans-serif", 30))
                    .margin(15)
                    .x_label_area_size(50)
                    .y_label_area_size(70)
                    .build_cartesian_2d(time_min..time_max, final_y_min..final_y_max)?;

                chart.configure_mesh()
                    .x_desc("Time (s)")
                    .y_desc(format!("Axis {} PIDsum", axis_index))
                    .x_labels(10)
                    .y_labels(10)
                    .light_line_style(&WHITE.mix(0.7))
                    .draw()?;

                chart.draw_series(LineSeries::new(pid_output_data, &GREEN))?
                    .label("PIDsum (P+I+D)")
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

                chart.configure_series_labels()
                    .position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8))
                    .border_style(&BLACK)
                    .draw()?;

                println!("  Axis {} PIDsum plot saved as '{}'.", axis_index, output_file);
            }
        } // End scope for PIDsum Plot


        // --- 2. Calculate and Plot Setpoint vs PIDsum --- // Changed section title comment
        { // Scope for Setpoint vs PIDsum Plot
            println!("  Generating Setpoint vs PIDsum plot..."); // Changed message

            // First, check if the required setpoint header for this axis exists
            if !setpoint_header_found[axis_index] {
                println!("  Skipping Axis {} Setpoint vs PIDsum Plot: Missing 'setpoint[{}]' header.", axis_index, axis_index); // Changed message
                continue; // Skip to the next axis
            }

            // Prepare data: requires time, setpoint, AND P, I, D terms for PIDsum calculation
            let setpoint_vs_pidsum_data: Vec<(f64, f64, f64)> = all_log_data.iter().filter_map(|row| { // Renamed variable
                if let (Some(time), Some(setpoint), Some(p), Some(i), Some(d)) =
                    (row.time_sec, row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    Some((time, setpoint, p + i + d)) // (time, setpoint, pidsum)
                } else {
                    None
                }
            }).collect();

             if setpoint_vs_pidsum_data.is_empty() {
                println!("  Skipping Axis {} Setpoint vs PIDsum Plot: No rows found with complete data (time, setpoint[{}], P, I, D).", axis_index, axis_index); // Changed message
                continue; // Skip to the next axis
            }

            // --- Determine Plot Ranges ---
            let (time_min, time_max) = setpoint_vs_pidsum_data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));

            // Find min/max across *both* setpoint and pidsum
            let (y_min, y_max) = setpoint_vs_pidsum_data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| {
                    (min_y.min(*s).min(*p), max_y.max(*s).max(*p))
                });

            let y_range = (y_max - y_min).abs();
            let y_padding = if y_range < 1e-6 { 0.5 } else { y_range * 0.1 };
            let final_y_min = y_min - y_padding;
            let final_y_max = y_max + y_padding;


            // --- Setup Plot ---
            let output_file = format!("{}_axis{}_setpoint_vs_pidsum.png", root_name, axis_index); // Changed filename
            let root_area = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
            root_area.fill(&WHITE)?;

            let title = format!("Axis {} Setpoint vs PIDsum", axis_index); // Changed title

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
                .y_desc(format!("Axis {} Value", axis_index))
                .x_labels(10)
                .y_labels(10)
                .light_line_style(&WHITE.mix(0.7))
                .draw()?;

            // --- Draw Series ---

            // Draw Setpoint Line (Blue)
            chart.draw_series(LineSeries::new(
                setpoint_vs_pidsum_data.iter().map(|(t, s, _p)| (*t, *s)),
                &BLUE,
            ))?
            .label("Setpoint")
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE));

            // Draw Calculated PIDsum Line (Red) - using the same data source
            chart.draw_series(LineSeries::new(
                setpoint_vs_pidsum_data.iter().map(|(t, _s, p)| (*t, *p)), // Map to (time, pidsum)
                &RED,
            ))?
            .label("PIDsum (P+I+D)") // Changed label
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

            chart.configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;

            println!("  Axis {} setpoint vs pidsum plot saved as '{}'.", axis_index, output_file); // Changed message

        } // End scope for Setpoint vs PIDsum Plot

    } // End axis plotting loop

    Ok(())
}