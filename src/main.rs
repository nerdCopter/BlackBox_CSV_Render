use csv::ReaderBuilder; // For reading CSV files efficiently.
use plotters::prelude::*; // For creating plots and charts.
use std::error::Error; // Standard trait for error handling.
use std::env; // For accessing command-line arguments.
use std::path::Path; // For working with file paths.
use std::fs::File; // For file operations (opening files).
use std::io::BufReader; // For buffered reading, improving file I/O performance.

/// Structure to hold the relevant data extracted from a single row of the CSV log.
/// Uses `Option<f64>` to gracefully handle missing or unparseable values in the CSV.
#[derive(Debug, Default, Clone)]
struct LogRowData {
    time_sec: Option<f64>,        // Timestamp of the log entry, converted to seconds.
    p_term: [Option<f64>; 3],     // Proportional term for each axis (Roll, Pitch, Yaw). Assumes CSV header like "axisP[0]".
    i_term: [Option<f64>; 3],     // Integral term for each axis. Assumes CSV header like "axisI[0]".
    d_term: [Option<f64>; 3],     // Derivative term for each axis. Assumes CSV header like "axisD[0]".
    setpoint: [Option<f64>; 3],   // Target setpoint value for each axis. Assumes CSV header like "setpoint[0]".
}

fn main() -> Result<(), Box<dyn Error>> {
    // --- Argument Parsing ---
    // Get command-line arguments passed to the program.
    let args: Vec<String> = env::args().collect();
    // Expecting the program name and the input CSV file path.
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file.csv>", args[0]);
        std::process::exit(1); // Exit if the filename is not provided.
    }
    let input_file = &args[1]; // The second argument is the input file path.
    let input_path = Path::new(input_file);
    // Extract the base name of the input file (without extension) to use in output filenames.
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();

    // --- Header Definition and Index Mapping ---
    // Define the specific CSV headers we are interested in extracting data from.
    // The order here defines the internal `target_idx` used later for mapping.
    let target_headers = [
        // Time (Essential for X-axis)
        "time (us)",    // 0: Base time unit, converted to seconds later.
        // P Term Components (Essential for PIDsum plot)
        "axisP[0]",     // 1: P term for Axis 0 (e.g., Roll)
        "axisP[1]",     // 2: P term for Axis 1 (e.g., Pitch)
        "axisP[2]",     // 3: P term for Axis 2 (e.g., Yaw)
        // I Term Components (Essential for PIDsum plot)
        "axisI[0]",     // 4: I term for Axis 0
        "axisI[1]",     // 5: I term for Axis 1
        "axisI[2]",     // 6: I term for Axis 2
        // D Term Components (Essential for PIDsum plot, except AxisD[2] which is optional)
        "axisD[0]",     // 7: D term for Axis 0
        "axisD[1]",     // 8: D term for Axis 1
        "axisD[2]",     // 9: D term for Axis 2 - Considered optional; defaults to 0.0 if header is missing.
        // Setpoint Components (Essential for Setpoint vs PIDsum plot)
        "setpoint[0]",  // 10: Setpoint for Axis 0
        "setpoint[1]",  // 11: Setpoint for Axis 1
        "setpoint[2]",  // 12: Setpoint for Axis 2
    ];

    // Flags to track if specific optional or plot-specific headers are found in the CSV.
    let mut axis_d2_header_found = false; // Tracks if "axisD[2]" is present.
    let mut setpoint_header_found = [false; 3]; // Tracks if "setpoint[x]" is present for each axis.

    // Read the CSV header row and map the target headers to their actual column indices.
    let header_indices: Vec<Option<usize>> = { // Use a scope to limit the lifetime of the reader.
        let file = File::open(input_file)?; // Open the input CSV file.
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // Indicate that the first row is the header.
            .trim(csv::Trim::All) // Trim whitespace from headers and fields.
            .from_reader(BufReader::new(file)); // Use a buffered reader for efficiency.

        // Get the actual headers from the CSV file.
        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        // Create a mapping: `indices[target_header_index]` will contain `Some(csv_column_index)` if found, or `None`.
        let indices: Vec<Option<usize>> = target_headers
            .iter()
            .map(|&target_header| {
                // Find the position (column index) of the target header in the actual CSV header record.
                header_record.iter().position(|h| h == target_header)
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut essential_pid_headers_found = true; // Assume all essentials are found initially.

        // Check if essential headers for the PIDsum plot (time, P[0-2], I[0-2], D[0-1]) are present.
        for i in 0..=8 { // Check target indices 0 through 8.
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_pid_headers_found = false; // Mark as missing if any essential header is not found.
                    format!("Not Found (Essential for PIDsum Plot!)")
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }

        // Check the optional header 'axisD[2]' (target index 9) separately.
        let axis_d2_name = target_headers[9];
        let axis_d2_status = match indices[9] {
            Some(idx) => {
                axis_d2_header_found = true; // Set the flag if found.
                format!("Found at index {}", idx)
            }
            None => {
                // It's okay if not found, we'll default its value later.
                format!("Not Found (Optional for PIDsum plot, will default to 0.0)")
            }
        };
        println!("  '{}' (Target Index {}): {}", axis_d2_name, 9, axis_d2_status);

        // Check setpoint headers (target indices 10, 11, 12) required for the Setpoint vs PIDsum plot.
        for axis in 0..3 {
            let target_idx = 10 + axis; // Calculate the target index for setpoint[axis].
            let name = target_headers[target_idx];
            let status = match indices[target_idx] {
                 Some(idx) => {
                    setpoint_header_found[axis] = true; // Set the flag for this specific axis if found.
                    format!("Found at index {}", idx)
                 }
                 None => {
                    // This plot cannot be generated for this axis if the header is missing.
                    format!("Not Found (Essential for Setpoint vs PIDsum Plot Axis {})", axis)
                 }
            };
            println!("  '{}' (Target Index {}): {}", name, target_idx, status);
        }

        // If any essential header for the PIDsum plot is missing, report the error and exit.
        if !essential_pid_headers_found {
             // Construct a detailed error message listing the missing essential headers.
             let missing_essentials: Vec<String> = (0..=8)
                 .filter(|&i| indices[i].is_none()) // Find which essential indices are None.
                 .map(|i| format!("'{}'", target_headers[i])) // Format the names of missing headers.
                 .collect();
             return Err(format!("Error: Missing essential headers for PIDsum plot: {}. Aborting.", missing_essentials.join(", ")).into());
        }
        indices // Return the calculated mapping of target headers to CSV column indices.
    };

    // --- Data Reading and Storage ---
    // Vector to store the parsed data from each valid row of the CSV.
    let mut all_log_data: Vec<LogRowData> = Vec::new();

    println!("\nReading P/I/D term and Setpoint data from CSV...");
    { // Scope for the main data reading CSV reader.
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true) // Skip the header row during data reading.
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        // Iterate over each row (record) in the CSV file.
        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    // Create a temporary structure to hold data for the current row.
                    let mut current_row_data = LogRowData::default();

                    // Helper closure to parse a value from the record based on the *target* header index.
                    // It uses the `header_indices` map to find the actual CSV column index.
                    let parse_f64_by_target_idx = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx) // Get the Option<usize> for the target index.
                            .and_then(|opt_csv_idx| opt_csv_idx.as_ref()) // Convert Option<usize> to Option<&usize>.
                            .and_then(|&csv_idx| record.get(csv_idx)) // Get the string value from the record using the CSV index.
                            .and_then(|val_str| val_str.parse::<f64>().ok()) // Attempt to parse the string as f64.
                    };

                    // --- Parse Time ---
                    // Get time (us) using the target index 0.
                    let time_us = parse_f64_by_target_idx(0);
                    if let Some(t_us) = time_us {
                         // Convert microseconds to seconds.
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         // Time is essential for plotting; skip row if missing or invalid.
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue; // Skip to the next row.
                    }

                    // --- Parse P, I, D Terms and Setpoint for each axis (0, 1, 2) ---
                    for axis in 0..3 {
                        // Parse P term for the current axis (target indices 1, 2, 3).
                        current_row_data.p_term[axis] = parse_f64_by_target_idx(1 + axis);
                        // Parse I term for the current axis (target indices 4, 5, 6).
                        current_row_data.i_term[axis] = parse_f64_by_target_idx(4 + axis);

                        // Parse D term for the current axis (target indices 7, 8, 9).
                        let d_target_idx = 7 + axis;
                        // Special handling for axis 2 D term: if header wasn't found, default to 0.0.
                        if axis == 2 && !axis_d2_header_found {
                             current_row_data.d_term[axis] = Some(0.0);
                        } else {
                             // Otherwise, parse normally.
                             current_row_data.d_term[axis] = parse_f64_by_target_idx(d_target_idx);
                        }

                        // Parse Setpoint for the current axis (target indices 10, 11, 12).
                        let setpoint_target_idx = 10 + axis;
                        // Only attempt to parse if the corresponding setpoint header was found earlier.
                        if setpoint_header_found[axis] {
                            current_row_data.setpoint[axis] = parse_f64_by_target_idx(setpoint_target_idx);
                        }
                        // If header wasn't found, `current_row_data.setpoint[axis]` remains None (from default).
                    }

                    // Add the parsed data for this row to the main data vector.
                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    // Report errors during CSV parsing but continue with the next row.
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // Reader goes out of scope here, and the input file is closed.

    println!("Finished reading {} data rows.", all_log_data.len());
    // Inform the user if axisD[2] was defaulted.
    if !axis_d2_header_found {
        println!("INFO: 'axisD[2]' header was not found. Used 0.0 as default value for Axis 2 D-term calculation in PIDsum plot.");
    }
    // Inform the user if setpoint plots cannot be generated due to missing headers.
    for axis in 0..3 {
        if !setpoint_header_found[axis] {
             println!("INFO: 'setpoint[{}]' header was not found. Setpoint vs PIDsum plot for Axis {} cannot be generated.", axis, axis);
        }
    }

    // If no valid data rows were read (e.g., only header or all rows had parsing errors), exit early.
    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }


    // --- Plotting Loop for Each Axis ---
    println!("\n--- Generating Plots for Each Axis ---");

    // Generate plots for each of the three axes (0, 1, 2).
    for axis_index in 0..3 {
        println!("Processing Axis {}...", axis_index);

        // --- 1. Calculate and Plot PIDsum (P+I+D vs Time) ---
        { // Scope specifically for the PIDsum plot generation.
            println!("  Generating PIDsum plot (P+I+D vs Time)...");

            // Filter and transform the raw data into plot points (time, pid_sum).
            // Only include rows where time, P, I, and D terms for the current axis are all present (Some).
            let pid_output_data: Vec<(f64, f64)> = all_log_data.iter().filter_map(|row| {
                if let (Some(time), Some(p), Some(i), Some(d)) =
                    // Ensure all necessary components for this axis exist in the row data.
                    (row.time_sec, row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    // Calculate the sum P + I + D.
                    Some((time, p + i + d))
                } else {
                    // If any component is missing (None), exclude this row from the plot data.
                    None
                }
            }).collect();

            // Check if we have any valid data points to plot for PIDsum.
            if pid_output_data.is_empty() {
                println!("  Skipping Axis {} PIDsum Plot: No rows found with complete P, I, and D term data.", axis_index);
            } else {
                // --- Determine Plot Ranges ---
                // Find the minimum and maximum time values for the X-axis range.
                let (time_min, time_max) = pid_output_data.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
                // Find the minimum and maximum PIDsum values for the Y-axis range.
                let (output_min, output_max) = pid_output_data.iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));
                // Calculate padding for the Y-axis to prevent data points touching the border.
                let y_range = (output_max - output_min).abs();
                let y_padding = if y_range < 1e-6 { 0.5 } else { y_range * 0.1 }; // Add small fixed padding if range is near zero.
                let final_y_min = output_min - y_padding;
                let final_y_max = output_max + y_padding;

                // --- Setup Plot ---
                let output_file = format!("{}_axis{}_PIDsum.png", root_name, axis_index);
                // Create a drawing backend targeting a PNG file.
                let root_area = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
                root_area.fill(&WHITE)?; // Fill the background with white.

                // Configure the chart title and margins.
                let title = format!("Axis {} PIDsum (P+I+D)", axis_index);
                let mut chart = ChartBuilder::on(&root_area)
                    .caption(title, ("sans-serif", 30))
                    .margin(15) // Margin around the plot area.
                    .x_label_area_size(50) // Space for X-axis labels.
                    .y_label_area_size(70) // Space for Y-axis labels and title.
                    .build_cartesian_2d(
                        time_min..time_max,         // X-axis range (time).
                        final_y_min..final_y_max    // Y-axis range (PIDsum with padding).
                    )?;

                // Configure the appearance of the axis labels and grid lines.
                chart.configure_mesh()
                    .x_desc("Time (s)") // X-axis label text.
                    .y_desc(format!("Axis {} PIDsum", axis_index)) // Y-axis label text.
                    .x_labels(10) // Number of labels on the X-axis.
                    .y_labels(10) // Number of labels on the Y-axis.
                    .light_line_style(&WHITE.mix(0.7)) // Style for light grid lines.
                    .draw()?; // Draw the mesh (grid and labels).

                // --- Draw Series ---
                // Draw the PIDsum data as a green line series.
                chart.draw_series(LineSeries::new(pid_output_data, &GREEN))?
                    .label("PIDsum (P+I+D)") // Label for the legend.
                    // Define how the legend marker should look (a short green line).
                    .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &GREEN));

                // Configure and draw the legend.
                chart.configure_series_labels()
                    .position(SeriesLabelPosition::UpperRight) // Place legend in the top-right.
                    .background_style(&WHITE.mix(0.8)) // Semi-transparent white background.
                    .border_style(&BLACK) // Black border around the legend.
                    .draw()?;

                println!("  Axis {} PIDsum plot saved as '{}'.", axis_index, output_file);
            }
        } // End scope for PIDsum Plot


        // --- 2. Calculate and Plot Setpoint vs PIDsum ---
        { // Scope specifically for the Setpoint vs PIDsum plot generation.
            println!("  Generating Setpoint vs PIDsum plot...");

            // First, check if the required setpoint header for this specific axis was found earlier.
            if !setpoint_header_found[axis_index] {
                println!("  Skipping Axis {} Setpoint vs PIDsum Plot: Missing 'setpoint[{}]' header.", axis_index, axis_index);
                continue; // Skip the rest of the plotting logic for this axis and move to the next axis.
            }

            // Prepare data: Filter rows where time, setpoint, P, I, AND D terms for the current axis are all present.
            // We need P, I, D to recalculate the PIDsum for comparison on the same plot.
            let setpoint_vs_pidsum_data: Vec<(f64, f64, f64)> = all_log_data.iter().filter_map(|row| { // (time, setpoint, pidsum)
                if let (Some(time), Some(setpoint), Some(p), Some(i), Some(d)) =
                    (row.time_sec, row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    // Store time, the setpoint value, and the calculated PID sum.
                    Some((time, setpoint, p + i + d))
                } else {
                    // Exclude rows missing any of these required components.
                    None
                }
            }).collect();

             // Check if we have any valid data points to plot after filtering.
             if setpoint_vs_pidsum_data.is_empty() {
                println!("  Skipping Axis {} Setpoint vs PIDsum Plot: No rows found with complete data (time, setpoint[{}], P, I, D).", axis_index, axis_index);
                continue; // Skip to the next axis.
            }

            // --- Determine Plot Ranges ---
            // Find the min/max time values for the X-axis (same logic as before).
            let (time_min, time_max) = setpoint_vs_pidsum_data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));

            // Find min/max Y values by considering *both* the setpoint and the PIDsum values in the data.
            // This ensures the Y-axis range encompasses both lines.
            let (y_min, y_max) = setpoint_vs_pidsum_data.iter()
                .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| {
                    // Update min_y with the minimum of current min_y, setpoint (s), and pidsum (p).
                    // Update max_y with the maximum of current max_y, setpoint (s), and pidsum (p).
                    (min_y.min(*s).min(*p), max_y.max(*s).max(*p))
                });

            // Calculate Y-axis padding based on the combined range of setpoint and PIDsum.
            let y_range = (y_max - y_min).abs();
            let y_padding = if y_range < 1e-6 { 0.5 } else { y_range * 0.1 };
            let final_y_min = y_min - y_padding;
            let final_y_max = y_max + y_padding;


            // --- Setup Plot ---
            // Define the output filename for this plot.
            let output_file = format!("{}_axis{}_setpoint_vs_pidsum.png", root_name, axis_index);
            let root_area = BitMapBackend::new(&output_file, (1024, 768)).into_drawing_area();
            root_area.fill(&WHITE)?;

            // Configure the chart title and margins.
            let title = format!("Axis {} Setpoint vs PIDsum", axis_index);

            let mut chart = ChartBuilder::on(&root_area)
                .caption(title, ("sans-serif", 30))
                .margin(15)
                .x_label_area_size(50)
                .y_label_area_size(70)
                .build_cartesian_2d(
                    time_min..time_max,         // X-axis range (time).
                    final_y_min..final_y_max,   // Y-axis range (encompassing setpoint and PIDsum).
                )?;

            // Configure the mesh (grid and labels).
            chart.configure_mesh()
                .x_desc("Time (s)")
                .y_desc(format!("Axis {} Value", axis_index)) // Generic Y-axis label as it shows two different values.
                .x_labels(10)
                .y_labels(10)
                .light_line_style(&WHITE.mix(0.7))
                .draw()?;

            // --- Draw Series ---

            // Draw Setpoint Line (Blue)
            // Map the collected data to (time, setpoint) tuples for plotting.
            chart.draw_series(LineSeries::new(
                setpoint_vs_pidsum_data.iter().map(|(t, s, _p)| (*t, *s)), // Extract (time, setpoint).
                &BLUE, // Use blue color for the setpoint line.
            ))?
            .label("Setpoint") // Legend label.
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE)); // Blue legend marker.

            // Draw Calculated PIDsum Line (Red)
            // Map the collected data to (time, pidsum) tuples for plotting.
            chart.draw_series(LineSeries::new(
                setpoint_vs_pidsum_data.iter().map(|(t, _s, p)| (*t, *p)), // Extract (time, pidsum).
                &RED, // Use red color for the PIDsum line.
            ))?
            .label("PIDsum (P+I+D)") // Legend label.
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED)); // Red legend marker.

            // Configure and draw the legend for both series.
            chart.configure_series_labels()
                .position(SeriesLabelPosition::UpperRight)
                .background_style(&WHITE.mix(0.8))
                .border_style(&BLACK)
                .draw()?;

            println!("  Axis {} setpoint vs pidsum plot saved as '{}'.", axis_index, output_file);

        } // End scope for Setpoint vs PIDsum Plot

    } // End axis plotting loop

    Ok(()) // Indicate successful execution.
}