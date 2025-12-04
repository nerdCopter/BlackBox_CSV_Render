// src/lib.rs - Library interface for internal module access

#![allow(non_snake_case)]

pub mod axis_names;
pub mod constants;
pub mod data_analysis;
pub mod data_input;
pub mod font_config;
pub mod pid_context;
pub mod plot_framework;
pub mod plot_functions;
pub mod types;

// Expose crate version derived from vergen-generated env vars at compile time.
pub fn crate_version() -> &'static str {
    option_env!("VERGEN_GIT_SEMVER").unwrap_or(env!("CARGO_PKG_VERSION"))
}
