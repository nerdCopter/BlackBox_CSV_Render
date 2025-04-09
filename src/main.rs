use csv::ReaderBuilder;
use plotters::prelude::*;
use std::error::Error;
use std::env;
use std::path::Path;

fn main() -> Result<(), Box<dyn Error>> {
    // Get the input file from command-line arguments
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: {} <input_file>", args[0]);
        std::process::exit(1);
    }
    let input_file = &args[1];

    // Derive the root name of the input file
    let input_path = Path::new(input_file);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();


    // Open the CSV file
    let file = std::fs::File::open(input_file)?;
    let mut reader = ReaderBuilder::new()
        .has_headers(true) // Assume the file has headers
        .from_reader(file);

    // Extract headers
    let headers = reader.headers()?.clone();
    println!("Headers: {:?}", headers);

    // List of headers we are interested in
    let target_headers = [
        "time (us)",
        "axisP[0]", "axisP[1]", "axisP[2]", 
        "axisI[0]", "axisI[1]", "axisI[2]", 
        "axisD[0]", "axisD[1]", "axisD[2]",
    ];

    // Find indices of target headers in the CSV file (trim spaces for matching)
    let header_indices: Vec<Option<usize>> = target_headers
        .iter()
        .map(|&header| {
            headers.iter().position(|h| h.trim() == header)
        })
        .collect();

    println!("Target header indices: {:?}", header_indices);

    for axis_index in 0..3 {
        //Prepare output filename for this axis
        let output_file = format!("{}_axis{}_step_response.png", root_name, axis_index);

        // Simulation parameters - these could be read from a config or command line
        let set_point = 10.0; // Example setpoint
        let dt = 0.1; // Time step - keep this consistent


        let mut time_data: Vec<f64> = Vec::new();
        let mut response_data: Vec<f64> = Vec::new();


             // Reset the CSV reader to start from the beginning for each axis
            let file = std::fs::File::open(input_file)?;
            let mut reader = ReaderBuilder::new()
                .has_headers(true)
                .from_reader(file);



        // initial values for simulation
        let mut previous_error = 0.0;
        let mut integral = 0.0;
        let mut process_variable = 0.0; //start at zero.
        let mut time = 0.0;



        while let Some(record) = reader.records().next() {
                let record = record?;

                let time_idx = header_indices[0];
                let p_idx = header_indices[1 + axis_index];
                let i_idx = header_indices[4 + axis_index];
                let d_idx = header_indices[7 + axis_index];



            if let (Some(time_idx_unwrapped),Some(p_idx_unwrapped), Some(i_idx_unwrapped), Some(d_idx_unwrapped)) = (time_idx, p_idx, i_idx, d_idx){
                   if let (Some(time_str), Some(p_str), Some(i_str), Some(d_str)) = (record.get(time_idx_unwrapped), record.get(p_idx_unwrapped), record.get(i_idx_unwrapped), record.get(d_idx_unwrapped)){

                        let time_us = time_str.trim().parse::<f64>().unwrap_or(0.0);
                        let kp = p_str.trim().parse::<f64>().unwrap_or(0.0);
                        let ki = i_str.trim().parse::<f64>().unwrap_or(0.0);
                        let kd = d_str.trim().parse::<f64>().unwrap_or(0.0);



                       let error = set_point - process_variable;

                       integral += error * dt;
                       let derivative = (error - previous_error) / dt;
                       let output = kp * error + ki * integral + kd * derivative;

                       process_variable += output * dt;


                        time_data.push(time);
                        response_data.push(process_variable);
                       previous_error = error;
                      time += dt;
                  }
            }
          }


        // Plotting the step response for this axis
        let root_area = BitMapBackend::new(&output_file, (800, 600)).into_drawing_area();
        root_area.fill(&WHITE).unwrap();

        let mut chart = ChartBuilder::on(&root_area)
            .caption(format!("Axis {} Step Response", axis_index), ("sans-serif", 50))
            .x_label_area_size(40)
            .y_label_area_size(40)
            .build_cartesian_2d(
                0.0..time_data.last().copied().unwrap_or(10.0), // Adjust x-axis range
                0.0..response_data.iter().copied().fold(0.0, f64::max) * 1.1, // Adjust y-axis range
            )
            .unwrap();

        chart.configure_mesh().draw().unwrap();

        chart.draw_series(LineSeries::new(
            time_data.into_iter().zip(response_data),
            &RED,
        ))
        .unwrap()
        .label("Step Response")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], &RED));

        chart.configure_series_labels()
            .background_style(&WHITE)
            .border_style(&BLACK)
            .draw()
            .unwrap();

        println!("Axis {} step response plot saved as '{}'.", axis_index, output_file);
    }

    Ok(())
}
