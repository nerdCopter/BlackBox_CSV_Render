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
    // These strings MUST match the *trimmed* headers in the CSV
    let target_headers = [
        // Time (Essential for X-axis)
        "time (us)",    // 0
        // P Term Components
        "axisP[0]",     // 1
        "axisP[1]",     // 2
        "axisP[2]",     // 3
        // I Term Components
        "axisI[0]",     // 4
        "axisI[1]",     // 5
        "axisI[2]",     // 6
        // D Term Components
        "axisD[0]",     // 7
        "axisD[1]",     // 8
        "axisD[2]",     // 9
    ];

    let header_indices = { // Scope for header reading
        let file = File::open(input_file)?;
        // Configure reader to trim whitespace from headers and fields
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All) // Crucial: trims whitespace from headers AND fields
            .from_reader(BufReader::new(file));

        // Get the headers (already trimmed by the ReaderBuilder)
        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV (trimmed): {:?}", header_record);

        // Find the indices by comparing target_headers to the trimmed header_record
        let indices: Vec<Option<usize>> = target_headers
            .iter()
            // No need for .trim() here anymore, ReaderBuilder did it
            .map(|&target_header| {
                header_record.iter().position(|h| h == target_header)
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut essential_headers_found = true;
        // Check essential headers (indices 0-9)
        for i in 0..=9 {
            let name = target_headers[i];
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    essential_headers_found = false;
                    format!("'{}' Not Found (Essential!)", name) // More specific error
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }

        if !essential_headers_found {
             // Construct a more detailed error message
             let missing_headers: Vec<String> = target_headers.iter().enumerate()
                 .filter_map(|(i, &name)| {
                     if i <= 9 && indices[i].is_none() { Some(format!("'{}'", name)) } else { None }
                 })
                 .collect();
             return Err(format!("Error: Missing essential headers needed for PID output calculation: {}. Please check CSV file and target_headers array. Aborting.", missing_headers.join(", ")).into());
        }
        indices // Return indices
    };

    // --- Data Reading and Storage ---
    let mut all_log_data: Vec<LogRowData> = Vec::new();

    println!("\nReading P/I/D term data from CSV...");
    { // Scope for the reader
        let file = File::open(input_file)?;
        // Use the same trimming settings for reading data rows
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All) // Ensure fields are also trimmed
            .from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut current_row_data = LogRowData::default();

                    // Helper to parse f64 field safely
                    let mut parse_f64 = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx) // Use .get() for safety
                            .and_then(|opt_idx| opt_idx.as_ref()) // Get Option<&usize> if Some
                            .and_then(|&csv_idx| record.get(csv_idx)) // Get Option<&str> field value
                            // No need to trim here, ReaderBuilder already did
                            .and_then(|val_str| val_str.parse::<f64>().ok()) // Parse
                    };

                    // --- Parse Time ---
                    let time_us = parse_f64(0); // Target index 0 = "time (us)"
                    if let Some(t_us) = time_us {
                         current_row_data.time_sec = Some(t_us / 1_000_000.0);
                    } else {
                         eprintln!("Warning: Skipping row {} due to missing or invalid 'time (us)'", row_index + 1);
                         continue; // Skip entire row if time is missing
                    }

                    // --- Parse P, I, D Terms for each axis ---
                    for axis in 0..3 {
                        // P term indices: 1, 2, 3
                        current_row_data.p_term[axis] = parse_f64(1 + axis);
                        // I term indices: 4, 5, 6
                        current_row_data.i_term[axis] = parse_f64(4 + axis);
                        // D term indices: 7, 8, 9
                        current_row_data.d_term[axis] = parse_f64(7 + axis);
                    }

                    // Add the fully parsed row data to our main vector
                    all_log_data.push(current_row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // Reader goes out of scope, file is closed

    println!("Finished reading {} data rows.", all_log_data.len());

    if all_log_data.is_empty() {
        println!("No valid data rows read, cannot generate plots.");
        return Ok(());
    }


    // --- Plotting Calculated PID Output ---
    println!("\n--- Generating Calculated PID Output Plots (P+I+D vs Time) ---");

    for axis_index in 0..3 {
        println!("Processing Axis {}...", axis_index);

        // Calculate PID output (P+I+D) for each time step where all components are available
        let pid_output_data: Vec<(f64, f64)> = all_log_data.iter().filter_map(|row| {
            // Ensure we have time and all three terms for this axis
            if let (Some(time), Some(p), Some(i), Some(d)) =
                (row.time_sec, row.p_term[axis_index], row.i_term[axis_index], row.d_term[axis_index])
            {
                Some((time, p + i + d)) // Calculate total PID output
            } else {
                None // Skip row if any component is missing
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