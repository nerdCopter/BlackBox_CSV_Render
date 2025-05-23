// src/log_parser.rs

use csv::{ReaderBuilder, StringRecord};
use std::collections::HashMap;
use std::error::Error;
use std::fs::File;
use std::io::Write;

use crate::log_data::LogRowData;

fn get_optional_f64(record: &StringRecord, index: Option<usize>) -> Option<f64> {
    index.and_then(|i| record.get(i).and_then(|s| s.trim().parse::<f64>().ok()))
}

fn get_f64_array_from_indices(record: &StringRecord, indices: &[Option<usize>; 3]) -> [Option<f64>; 3] {
    [
        get_optional_f64(record, indices[0]),
        get_optional_f64(record, indices[1]),
        get_optional_f64(record, indices[2]),
    ]
}

fn get_f64_debug_array(record: &StringRecord, indices: &[Option<usize>; 4]) -> [Option<f64>; 4] {
    [
        get_optional_f64(record, indices[0]),
        get_optional_f64(record, indices[1]),
        get_optional_f64(record, indices[2]),
        get_optional_f64(record, indices[3]),
    ]
}

pub fn parse_csv(
    filename: &str,
    mut diag_file: Option<&mut File>,
) -> Result<(Vec<LogRowData>, Option<f64>, Vec<String>), Box<dyn Error>> {
    let mut rdr = ReaderBuilder::new().has_headers(true).from_path(filename)?;
    let headers_record = rdr.headers()?.clone();
    let headers: Vec<String> = headers_record.iter().map(|s| s.to_string()).collect();

    if let Some(file) = diag_file.as_mut() {
        writeln!(file, "Headers found in CSV: {:?}", headers_record)?;
    } else {
        println!("Headers found in CSV: {:?}", headers_record);
    }


    let mut column_indices: HashMap<String, usize> = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        column_indices.insert(header.trim().to_string(), i);
    }

    let mut log_data_vec: Vec<LogRowData> = Vec::new();
    let mut time_values: Vec<f64> = Vec::new();

    // Define desired headers and their importance/fallback logic
    let p_indices = [
        column_indices.get("axisP[0]").copied(),
        column_indices.get("axisP[1]").copied(),
        column_indices.get("axisP[2]").copied(),
    ];
    let i_indices = [
        column_indices.get("axisI[0]").copied(),
        column_indices.get("axisI[1]").copied(),
        column_indices.get("axisI[2]").copied(),
    ];
    // Allow D term for Yaw (axisD[2]) to be optional for some flight controllers
    let d_indices = [
        column_indices.get("axisD[0]").copied(),
        column_indices.get("axisD[1]").copied(),
        column_indices.get("axisD[2]").copied(), // Optional
    ];
    let setpoint_indices = [
        column_indices.get("setpoint[0]").copied(),
        column_indices.get("setpoint[1]").copied(),
        column_indices.get("setpoint[2]").copied(),
    ];
    let gyro_adc_indices = [
        column_indices.get("gyroADC[0]").copied(),
        column_indices.get("gyroADC[1]").copied(),
        column_indices.get("gyroADC[2]").copied(),
    ];
    let gyro_unfilt_indices = [
        column_indices.get("gyroUnfilt[0]").copied(),
        column_indices.get("gyroUnfilt[1]").copied(),
        column_indices.get("gyroUnfilt[2]").copied(),
    ];
     let debug_indices = [
        column_indices.get("debug[0]").copied(),
        column_indices.get("debug[1]").copied(),
        column_indices.get("debug[2]").copied(),
        column_indices.get("debug[3]").copied(),
     ];
     let time_index = column_indices.get("time (us)").copied();
     let throttle_index = column_indices.get("debug[3]").copied();     // Throttle from debug[3]

     // ---------- Validate essential headers ----------
     let mut missing_essential: Vec<&str> = Vec::new();
     for (name, idx) in [
        ("time (us)", time_index),
        ("setpoint[0]", setpoint_indices[0]),
        ("setpoint[1]", setpoint_indices[1]),
        ("setpoint[2]", setpoint_indices[2]),
        ("setpoint[3]", throttle_index),
        ("gyroADC[0]", gyro_adc_indices[0]),
        ("gyroADC[1]", gyro_adc_indices[1]),
        ("gyroADC[2]", gyro_adc_indices[2]),
     ] {
        // ---------- Validate essential headers ----------
        let mut missing_essential: Vec<&str> = Vec::new();
        for (name, idx) in [
        ("time (us)", time_index),
        ("setpoint[0]", setpoint_indices[0]),
        ("setpoint[1]", setpoint_indices[1]),
        ("setpoint[2]", setpoint_indices[2]),
        ("setpoint[3]", throttle_index),
        ("gyroADC[0]", gyro_adc_indices[0]),
        ("gyroADC[1]", gyro_adc_indices[1]),
        ("gyroADC[2]", gyro_adc_indices[2]),
        ] {

        // Fallback for gyroUnfilt using debug fields if gyroUnfilt is missing
        for i in 0..3 {
            if gyro_unfilt_values[i].is_none() {
                gyro_unfilt_values[i] = debug_values[i];
            }
        }
        
        let throttle_value = get_optional_f64(&record, throttle_index);

        if let Some(t_us) = time_us {
            let time_s = t_us / 1_000_000.0;
            time_values.push(time_s);
            log_data_vec.push(LogRowData {
                time_sec: Some(time_s),
                p_term: p_terms,
                i_term: i_terms,
                d_term: d_terms,
                setpoint: setpoints,
                gyro: gyro_adc_values,
                gyro_unfilt: gyro_unfilt_values,
                debug: debug_values,
                throttle: throttle_value,
            });
        }
    }

    let sample_rate_option = if time_values.len() > 1 {
        let total_time = time_values.last().unwrap() - time_values.first().unwrap();
        if total_time > std::f64::EPSILON {
            Some((time_values.len() - 1) as f64 / total_time)
        } else {
            None  // or return an Err(...) explaining corrupted timestamps
        }
    } else {
        None
    };

    Ok((log_data_vec, sample_rate_option, headers))
}

// src/log_parser.rs
