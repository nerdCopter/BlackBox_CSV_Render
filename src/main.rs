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

    // Derive the output file name from the input file
    let input_path = Path::new(input_file);
    let root_name = input_path.file_stem().unwrap_or_default().to_string_lossy();
    let output_file = format!("{}_step_responses.png", root_name);

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

    // Data storage for plotting each axis response
    let mut axis_responses: Vec<Vec<(f64, f64)>> = Vec::new();

   for axis_index in 0..3 {
        let mut data_points: Vec<(f64, f64)> = Vec::new();

        let time_index = header_indices[0];
        let p_index = header_indices[1 + axis_index];
        let i_index = header_indices[4 + axis_index];
        let d_index = header_indices[7 + axis_index];
     // Reset the CSV reader to start from the beginning
        let file = std::fs::File::open(input_file)?;
         let mut reader = ReaderBuilder::new()
            .has_headers(true) // Assume the file has headers
            .from_reader(file);

        for record in reader.records() {
            let record = record?;

                if let (Some(time_idx), Some(p_idx), Some(i_idx), Some(d_idx)) = (time_index, p_index, i_index, d_index){
                    if let (Some(time_str), Some(p_str), Some(i_str), Some(d_str)) = (record.get(time_idx), record.get(p_idx), record.get(i_idx), record.get(d_idx)){

                        let time_us = time_str.trim().parse::<f64>().unwrap_or(0.0);
                        let p = p_str.trim().parse::<f64>().unwrap_or(0.0);
                        let i = i_str.trim().parse::<f64>().unwrap_or(0.0);
                        let d = d_str.trim().parse::<f64>().unwrap_or(0.0);

                        let output = p + i + d; // approximate pid output

                        data_points.push((time_us / 1_000_000.0, output)); // time in seconds
                    }

                }


        }
        axis_responses.push(data_points);



    }


    // Plotting the responses for all axes and components
    let root_area = BitMapBackend::new(&output_file, (1200, 800)).into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_area)
        .caption("PID Responses", ("sans-serif", 50))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(
             0.0..axis_responses.iter()
             .flat_map(|data| data.iter().map(|(time, _)| time))
            .fold(0.0, |acc, &x| f64::max(acc, x)), // Adjust x-axis range
        0.0..axis_responses.iter()
                .flat_map(|data| data.iter().map(|(_, value)| value))
                .fold(0.0, |acc, &x| f64::max(acc, x)) * 1.1
                
        )
        .unwrap();

    chart.configure_mesh().draw().unwrap();

    let colors = [&RED, &GREEN, &BLUE]; // Define colors for each component

    for (i, data_points) in axis_responses.into_iter().enumerate() {
          chart.draw_series(LineSeries::new(
                data_points.into_iter(),
                colors[i % 3],
            ))
            .unwrap()
            .label(format!("Axis {}", i))
            .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], colors[i % 3]));
    }

    chart.configure_series_labels()
        .background_style(&WHITE)
        .border_style(&BLACK)
        .draw()
        .unwrap();

    println!("PID responses plot saved as '{}'.", output_file);

    Ok(())
}
