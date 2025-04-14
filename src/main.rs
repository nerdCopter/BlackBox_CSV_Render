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

// Define constants for plot dimensions
const PLOT_WIDTH: u32 = 1920;
const PLOT_HEIGHT: u32 = 1080;

// Helper function to calculate plot range with padding
fn calculate_range(min_val: f64, max_val: f64) -> (f64, f64) {
    let range = (max_val - min_val).abs();
    // Add a slightly larger padding for potentially smaller subplot value ranges
    let padding = if range < 1e-6 { 0.5 } else { range * 0.15 };
    (min_val - padding, max_val + padding)
}

// Helper function to draw "Data Unavailable" message
fn draw_unavailable_message(
    area: &DrawingArea<BitMapBackend, plotters::coord::Shift>,
    axis_index: usize,
    plot_type: &str,
) -> Result<(), Box<dyn Error>> {
    let message = format!("Axis {} {} Data Unavailable", axis_index, plot_type);
    area.draw(&Text::new(
        message,
        (50, 50), // Position of the text within the subplot
        ("sans-serif", 20).into_font().color(&RED), // Style
    ))?;
    Ok(())
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

    // --- Data Preparation for Plots ---
    // These vectors will hold the plot data specific to each axis.
    let mut pid_output_data: [Vec<(f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    let mut setpoint_vs_pidsum_data: [Vec<(f64, f64, f64)>; 3] = [Vec::new(), Vec::new(), Vec::new()];
    // Flags to track if data is actually available to plot for each axis/type.
    let mut pid_data_available = [false; 3];
    let mut setpoint_data_available = [false; 3];

    // Iterate through the collected log data once to populate data for all axes
    for row in &all_log_data {
        if let Some(time) = row.time_sec {
            for axis_index in 0..3 {
                // PIDsum data
                if let (Some(p), Some(i), Some(d)) =
                    (row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                {
                    pid_output_data[axis_index].push((time, p + i + d));
                    if !pid_data_available[axis_index] { // Only set true once
                         pid_data_available[axis_index] = true;
                    }
                }

                // Setpoint vs PIDsum data
                if setpoint_header_found[axis_index] { // Only process if setpoint header exists
                    if let (Some(setpoint), Some(p), Some(i), Some(d)) =
                        (row.setpoint[axis_index], row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
                    {
                        setpoint_vs_pidsum_data[axis_index].push((time, setpoint, p + i + d));
                         if !setpoint_data_available[axis_index] { // Only set true once
                            setpoint_data_available[axis_index] = true;
                         }
                    }
                }
            }
        }
    }

    // --- Generate Stacked PIDsum Plot ---
    println!("\n--- Generating Stacked PIDsum Plot (All Axes) ---");
    if pid_data_available.iter().any(|&x| x) { // Check if data exists for at least one axis
        let output_file_pidsum = format!("{}_PIDsum_stacked.png", root_name);
        let root_area_pidsum = BitMapBackend::new(&output_file_pidsum, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_pidsum.fill(&WHITE)?;

        // Split the drawing area into 3 vertical parts
        let sub_plot_areas = root_area_pidsum.split_evenly((3, 1)); // 3 rows, 1 column
        // Define the color for PIDsum plots outside the loop
        let pidsum_plot_color = Palette99::pick(1); // Use Green consistently

        for axis_index in 0..3 {
            let area = &sub_plot_areas[axis_index]; // Get the drawing area for the current axis

            if pid_data_available[axis_index] {
                // Determine ranges *specifically for this axis*
                let (time_min, time_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _)| (min_t.min(*t), max_t.max(*t)));
                let (output_min, output_max) = pid_output_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_v, max_v), (_, v)| (min_v.min(*v), max_v.max(*v)));

                // Handle case where axis has only one data point or no range
                 if time_min.is_infinite() || output_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "PIDsum")?;
                     continue;
                 }

                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_pidsum_min, final_pidsum_max) = calculate_range(output_min, output_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} PIDsum (P+I+D)", axis_index), ("sans-serif", 20)) // Smaller caption
                    .margin(5) // Reduced margin
                    .x_label_area_size(30) // Reduced label area
                    .y_label_area_size(50) // Reduced label area
                    .build_cartesian_2d(final_time_min..final_time_max, final_pidsum_min..final_pidsum_max)?;

                chart.configure_mesh()
                    .x_desc("Time (s)")
                    .y_desc("PIDsum")
                    .x_labels(10) // Fewer labels might be needed
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7))
                    .label_style(("sans-serif", 12)) // Smaller label font
                    .draw()?;

                chart.draw_series(LineSeries::new(
                    pid_output_data[axis_index].iter().cloned(),
                    &pidsum_plot_color, // Use the color defined outside
                ))?;
                // No legend needed as title indicates the content

            } else {
                println!("  INFO: No PIDsum data available for Axis {}. Drawing placeholder.", axis_index);
                draw_unavailable_message(area, axis_index, "PIDsum")?;
            }
        }
        root_area_pidsum.present()?;
        println!("  Stacked PIDsum plot saved as '{}'.", output_file_pidsum);

    } else {
        println!("  Skipping Stacked PIDsum Plot: No PIDsum data available for any axis.");
    }


    // --- Generate Stacked Setpoint vs PIDsum Plot ---
    println!("\n--- Generating Stacked Setpoint vs PIDsum Plot (All Axes) ---");
    if setpoint_data_available.iter().any(|&x| x) { // Check if data exists for at least one axis
        let output_file_setpoint = format!("{}_SetpointVsPIDsum_stacked.png", root_name);
        let root_area_setpoint = BitMapBackend::new(&output_file_setpoint, (PLOT_WIDTH, PLOT_HEIGHT)).into_drawing_area();
        root_area_setpoint.fill(&WHITE)?;

        // Split the drawing area into 3 vertical parts
        let sub_plot_areas = root_area_setpoint.split_evenly((3, 1)); // 3 rows, 1 column
        // Define colors outside the loop to ensure they live long enough
        let setpoint_plot_color = Palette99::pick(2); // Blue
        let pidsum_vs_setpoint_color = Palette99::pick(0); // Red

        for axis_index in 0..3 {
             let area = &sub_plot_areas[axis_index]; // Get the drawing area for the current axis

            if setpoint_data_available[axis_index] {
                // Determine ranges *specifically for this axis*
                let (time_min, time_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_t, max_t), (t, _, _)| (min_t.min(*t), max_t.max(*t)));
                let (val_min, val_max) = setpoint_vs_pidsum_data[axis_index].iter()
                    .fold((f64::INFINITY, f64::NEG_INFINITY), |(min_y, max_y), (_, s, p)| {
                        (min_y.min(*s).min(*p), max_y.max(*s).max(*p))
                    });

                 // Handle case where axis has only one data point or no range
                 if time_min.is_infinite() || val_min.is_infinite() {
                     draw_unavailable_message(area, axis_index, "Setpoint/PIDsum")?;
                     continue;
                 }

                let (final_time_min, final_time_max) = (time_min, time_max);
                let (final_value_min, final_value_max) = calculate_range(val_min, val_max);

                let mut chart = ChartBuilder::on(area)
                    .caption(format!("Axis {} Setpoint vs PIDsum", axis_index), ("sans-serif", 20))
                    .margin(5)
                    .x_label_area_size(30)
                    .y_label_area_size(50)
                    .build_cartesian_2d(final_time_min..final_time_max, final_value_min..final_value_max)?;

                chart.configure_mesh()
                    .x_desc("Time (s)")
                    .y_desc("Value")
                    .x_labels(10)
                    .y_labels(5)
                    .light_line_style(&WHITE.mix(0.7))
                    .label_style(("sans-serif", 12))
                    .draw()?;

                // Draw Setpoint Line (Blue)
                // Use the color defined outside the loop
                let sp_color_ref = &setpoint_plot_color;
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, s, _p)| (*t, *s)),
                    sp_color_ref,
                ))?
                .label("Setpoint")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], sp_color_ref)); // Pass reference

                // Draw PIDsum Line (Red)
                // Use the color defined outside the loop
                let pid_color_ref = &pidsum_vs_setpoint_color;
                chart.draw_series(LineSeries::new(
                    setpoint_vs_pidsum_data[axis_index].iter().map(|(t, _s, p)| (*t, *p)),
                    pid_color_ref,
                ))?
                .label("PIDsum")
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], pid_color_ref)); // Pass reference

                chart.configure_series_labels()
                    .position(SeriesLabelPosition::UpperRight)
                    .background_style(&WHITE.mix(0.8))
                    .border_style(&BLACK)
                    .label_font(("sans-serif", 12)) // Smaller legend font
                    .draw()?;

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

    Ok(()) // Indicate successful execution.
}