use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;
use std::fs::File;
use std::io::BufReader;

// Helper struct to hold data for a single row
#[derive(Debug, Default, Clone)]
struct RowData {
    loop_iteration: Option<u64>,
    time_us: Option<f64>,
    axis_p: [Option<f64>; 3],
    axis_i: [Option<f64>; 3],
    axis_d: [Option<f64>; 3],
    setpoint: [Option<f64>; 3],
    gyro_adc: [Option<f64>; 3],     // New field
    gyro_unfilt: [Option<f64>; 3],  // New field
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
    // Headers we want to find and read
    let target_headers = [
        // Essential
        "loopIteration", // 0
        "time (us)",    // 1
        // Axis 0 PID + Setpoint
        "axisP[0]",     // 2
        "axisI[0]",     // 3
        "axisD[0]",     // 4
        "setpoint[0]",  // 5
        // Axis 1 PID + Setpoint
        "axisP[1]",     // 6
        "axisI[1]",     // 7
        "axisD[1]",     // 8
        "setpoint[1]",  // 9
        // Axis 2 PID + Setpoint
        "axisP[2]",     // 10
        "axisI[2]",     // 11
        "axisD[2]",     // 12
        "setpoint[2]",  // 13
        // Gyro ADC (New)
        "gyroADC[0]",   // 14
        "gyroADC[1]",   // 15
        "gyroADC[2]",   // 16
        // Gyro Unfiltered (New)
        "gyroUnfilt[0]",// 17
        "gyroUnfilt[1]",// 18
        "gyroUnfilt[2]",// 19
        // Optional setpoint[3]
        "setpoint[3]",  // 20
    ];

    let (_headers, header_indices) = { // Renamed headers to _headers as it's not used directly after this block
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file)); // Use BufReader

        let header_record = reader.headers()?.clone();
        println!("Headers found in CSV: {:?}", header_record);

        let indices: Vec<Option<usize>> = target_headers
            .iter()
            .map(|&target_header| {
                header_record.iter().position(|h| h == target_header)
            })
            .collect();

        println!("Indices map (Target Header -> CSV Index):");
        let mut all_found = true;
        for (i, name) in target_headers.iter().enumerate() {
             // Check if the header is one of the essential ones we absolutely need for printing/sim
             let is_essential = i <= 19; // Indices 0-19 are now used in printing
             let found_status = match indices[i] {
                 Some(idx) => format!("Found at index {}", idx),
                 None => {
                    if is_essential { all_found = false; } // Mark if an essential one is missing
                    "Not Found".to_string()
                 }
             };
             println!("  '{}' (Target Index {}): {}", name, i, found_status);
        }
        if !all_found {
             eprintln!("Warning: Not all target headers (up to index 19) were found. Data extraction and printing might be incomplete.");
        }
        (header_record, indices) // Return headers and indices
    };

    // --- Data Reading and Storage ---
    let mut all_data: Vec<RowData> = Vec::new();
    println!("\nReading data from CSV...");
    { // Scope for the reader
        let file = File::open(input_file)?;
        let mut reader = ReaderBuilder::new()
            .has_headers(true)
            .trim(csv::Trim::All)
            .from_reader(BufReader::new(file));

        for (row_index, result) in reader.records().enumerate() {
            match result {
                Ok(record) => {
                    let mut row_data = RowData::default();

                    // Helper to parse a value safely
                    let mut parse_field = |target_idx: usize| -> Option<f64> {
                        header_indices.get(target_idx) // Use get for safety
                            .and_then(|opt_idx| opt_idx.as_ref()) // Get Option<&usize>
                            .and_then(|&csv_idx| record.get(csv_idx)) // Get Option<&str>
                            .and_then(|val_str| val_str.parse::<f64>().ok()) // Parse
                    };
                    // Helper to parse u64
                     let mut parse_u64_field = |target_idx: usize| -> Option<u64> {
                        header_indices.get(target_idx)
                            .and_then(|opt_idx| opt_idx.as_ref())
                            .and_then(|&csv_idx| record.get(csv_idx))
                            .and_then(|val_str| val_str.parse::<u64>().ok())
                    };


                    // Extract data using helpers
                    row_data.loop_iteration = parse_u64_field(0); // target_headers[0] = "loopIteration"
                    row_data.time_us = parse_field(1);         // target_headers[1] = "time (us)"

                    for axis in 0..3 {
                        // PID + Setpoint Indices: P=2,6,10; I=3,7,11; D=4,8,12; SP=5,9,13
                        row_data.axis_p[axis] = parse_field(2 + axis * 4);
                        row_data.axis_i[axis] = parse_field(3 + axis * 4);
                        row_data.axis_d[axis] = parse_field(4 + axis * 4);
                        row_data.setpoint[axis] = parse_field(5 + axis * 4);

                        // Gyro Indices: ADC=14,15,16; Unfilt=17,18,19
                        row_data.gyro_adc[axis] = parse_field(14 + axis);   // New
                        row_data.gyro_unfilt[axis] = parse_field(17 + axis); // New
                    }

                    all_data.push(row_data);
                }
                Err(e) => {
                    eprintln!("Warning: Skipping row {} due to CSV read error: {}", row_index + 1, e);
                }
            }
        }
    } // Reader goes out of scope, file is closed

    println!("Finished reading {} data rows.", all_data.len());

    // --- Columnar Printing ---
    if !all_data.is_empty() {
        println!("\n--- Extracted Data ---");

        // Define column widths (adjust as needed)
        const W_ITER: usize = 12;
        const W_TIME: usize = 15;
        const W_PID: usize = 10;
        const W_SP: usize = 12;
        const W_GYRO: usize = 10; // Width for gyro values

        // Build Header String Dynamically (more maintainable)
        let header = format!(
            "{:<iter$} {:<time$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<gyro$} {:<gyro$} {:<gyro$} | {:<gyro$} {:<gyro$} {:<gyro$}",
            "LoopIter", "Time (us)",
            "P[0]", "I[0]", "D[0]", "SP[0]",
            "P[1]", "I[1]", "D[1]", "SP[1]",
            "P[2]", "I[2]", "D[2]", "SP[2]",
            "ADC[0]", "ADC[1]", "ADC[2]",       // New Headers
            "Unfilt[0]", "Unfilt[1]", "Unfilt[2]", // New Headers
            iter = W_ITER, time = W_TIME, pid = W_PID, sp = W_SP, gyro = W_GYRO
        );
        println!("{}", header);
        println!("{}", "-".repeat(header.len())); // Separator line matching header length

        // Print Data Rows
        for data in &all_data {
            // Helper to format Option<T>
            fn fmt_opt<T: std::fmt::Display>(opt: &Option<T>, default: &str) -> String {
                opt.as_ref().map_or(default.to_string(), |v| format!("{}", v))
            }
            fn fmt_opt_f64(opt: &Option<f64>, default: &str, precision: usize) -> String {
                opt.as_ref().map_or(default.to_string(), |v| format!("{:.prec$}", v, prec = precision))
            }

            // Build Data Row String Dynamically
             let data_row = format!(
                "{:<iter$} {:<time$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<pid$} {:<pid$} {:<pid$} {:<sp$} | {:<gyro$} {:<gyro$} {:<gyro$} | {:<gyro$} {:<gyro$} {:<gyro$}",
                fmt_opt(&data.loop_iteration, "N/A"),
                fmt_opt_f64(&data.time_us, "N/A", 1), // time with 1 decimal
                // Axis 0
                fmt_opt_f64(&data.axis_p[0], "N/A", 3),
                fmt_opt_f64(&data.axis_i[0], "N/A", 3),
                fmt_opt_f64(&data.axis_d[0], "N/A", 3),
                fmt_opt_f64(&data.setpoint[0], "N/A", 3),
                // Axis 1
                fmt_opt_f64(&data.axis_p[1], "N/A", 3),
                fmt_opt_f64(&data.axis_i[1], "N/A", 3),
                fmt_opt_f64(&data.axis_d[1], "N/A", 3),
                fmt_opt_f64(&data.setpoint[1], "N/A", 3),
                // Axis 2
                fmt_opt_f64(&data.axis_p[2], "N/A", 3),
                fmt_opt_f64(&data.axis_i[2], "N/A", 3),
                fmt_opt_f64(&data.axis_d[2], "N/A", 3),
                fmt_opt_f64(&data.setpoint[2], "N/A", 3),
                 // Gyro ADC (New)
                fmt_opt_f64(&data.gyro_adc[0], "N/A", 3),
                fmt_opt_f64(&data.gyro_adc[1], "N/A", 3),
                fmt_opt_f64(&data.gyro_adc[2], "N/A", 3),
                // Gyro Unfilt (New)
                fmt_opt_f64(&data.gyro_unfilt[0], "N/A", 3),
                fmt_opt_f64(&data.gyro_unfilt[1], "N/A", 3),
                fmt_opt_f64(&data.gyro_unfilt[2], "N/A", 3),
                // Width arguments
                iter = W_ITER, time = W_TIME, pid = W_PID, sp = W_SP, gyro = W_GYRO
            );
            println!("{}", data_row);
        }
        println!("{}\n", "-".repeat(header.len())); // Footer line
    } else {
        println!("No data read from CSV, skipping columnar print and simulation.");
        return Ok(()); // Exit early if no data
    }

    // --- Simulation and Plotting (Using FIRST row's data for PID parameters) ---
    println!("--- Starting Simulation Based on FIRST Data Row's PID Parameters ---");

    // Get PID parameters from the *first* row of the stored data
    let first_data_row = all_data.first().ok_or("No data rows available for simulation parameters")?;

    for axis_index in 0..3 {
        println!("\n--- Simulating Axis {} ---", axis_index);

        // Extract Kp, Ki, Kd, Setpoint from the first row's data for the specific axis
        // Use 0.0 as default if data wasn't present/parsable in the first row
        let kp = first_data_row.axis_p[axis_index].unwrap_or(0.0);
        let ki = first_data_row.axis_i[axis_index].unwrap_or(0.0);
        let kd = first_data_row.axis_d[axis_index].unwrap_or(0.0);
        let set_point = first_data_row.setpoint[axis_index].unwrap_or(0.0);

        println!("  Using parameters from first row for Axis {}:", axis_index);
        println!("    Kp: {}", kp);
        println!("    Ki: {}", ki);
        println!("    Kd: {}", kd);
        println!("    Setpoint: {}", set_point);

        // Prepare output filename
        let output_file = format!("{}_axis{}_simulated_step_response.png", root_name, axis_index);


        // --- Simulation part (unchanged logic) ---
        let mut time_data: Vec<f64> = Vec::new();
        let mut response_data: Vec<f64> = Vec::new();
        let dt = 0.001; // Simulation time step
        let mut previous_error = 0.0;
        let mut integral = 0.0;
        let mut process_variable = 0.0; // start at zero.
        let mut time = 0.0;
        let simulation_duration = 5.0; // Simulate for 5 seconds

        while time < simulation_duration {
            let error = set_point - process_variable;
            let proportional_term = kp * error;
            integral += ki * error * dt;
            let derivative_term = if dt > 0.0 { kd * (error - previous_error) / dt } else { 0.0 };
            let output = proportional_term + integral + derivative_term;
            process_variable += output * dt;

            time_data.push(time);
            response_data.push(process_variable);
            previous_error = error;
            time += dt;
        }

        // --- Plotting part (unchanged logic, except filename) ---
        let min_response = response_data.iter().copied().fold(f64::INFINITY, f64::min);
        let max_response = response_data.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let y_range_padding = (max_response - min_response).abs() * 0.1 + 0.1; // Add small base padding
        let y_min = min_response - y_range_padding;
        let y_max = max_response + y_range_padding;
        // Adjust y-axis range to potentially include 0 and the setpoint nicely
        let final_y_min = if set_point >= 0.0 && y_min > 0.0 { 0.0f64.min(y_min) } else { y_min };
        let final_y_max = if set_point > 0.0 { set_point.max(y_max) * 1.1 } else { y_max };


        let root_area = BitMapBackend::new(&output_file, (800, 600)).into_drawing_area();
        root_area.fill(&WHITE)?;

        let mut chart = ChartBuilder::on(&root_area)
            .caption(format!("Simulated Axis {} Step Response (Kp={}, Ki={}, Kd={}, SP={})", axis_index, kp, ki, kd, set_point), ("sans-serif", 30))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(
                0.0..simulation_duration,
                final_y_min..final_y_max,
            )?;

        chart.configure_mesh()
            .x_desc("Time (s)")
            .y_desc("Simulated Process Variable")
            .draw()?;

        chart.draw_series(LineSeries::new(
            vec![(0.0, set_point), (simulation_duration, set_point)],
            &BLUE.mix(0.5),
        ))?
        .label(format!("Setpoint ({})", set_point))
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &BLUE.mix(0.5)));


        chart.draw_series(LineSeries::new(
            time_data.into_iter().zip(response_data),
            &RED,
        ))?
        .label("Simulated Response")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels()
            .background_style(&WHITE.mix(0.8))
            .border_style(&BLACK)
            .draw()?;

        println!("Axis {} simulated step response plot saved as '{}'.", axis_index, output_file);

    } // End axis simulation loop

    Ok(())
}
