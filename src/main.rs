use plotters::prelude::*;

fn main() {
    // PID parameters
    let kp = 1.0; // Proportional gain
    let ki = 0.5; // Integral gain
    let kd = 0.1; // Derivative gain
    let set_point = 10.0; // Desired target value

    // Simulation parameters
    let mut process_variable = 0.0; // Initial system output
    let mut time = 0.0;
    let dt = 0.1; // Time step
    let simulation_duration = 10.0; // Total simulation time

    // Data storage for plotting
    let mut time_data = Vec::new();
    let mut response_data = Vec::new();

    while time <= simulation_duration {
        // Simulate a simple PID control (for demonstration purposes)
        let error = set_point - process_variable;
        let proportional = kp * error;
        let integral = ki * error * dt;
        let derivative = kd * (error / dt);
        let output = proportional + integral + derivative;

        process_variable += output * dt;

        // Store data for plotting
        time_data.push(time);
        response_data.push(process_variable);

        time += dt;
    }

    // Plotting the step response
    let root_area = BitMapBackend::new("step_response.png", (800, 600))
        .into_drawing_area();
    root_area.fill(&WHITE).unwrap();

    let mut chart = ChartBuilder::on(&root_area)
        .caption("PID Step Response", ("sans-serif", 50))
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_cartesian_2d(0.0..simulation_duration, 0.0..set_point * 2.0)
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
}
