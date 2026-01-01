## This is an experimental RUST program, mostly created via A.I., to read Betaflight Blackbox CSV and produce meaningful graphs.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Build](#build)
- [Usage](#usage)
- [Example execution commands](#example-execution-commands)
- [Output](#output)
- [Code Overview](#code-overview)
- [Development](#development)
  - [Setting Up Development Environment](#setting-up-development-environment)
  - [Required Commands Before Committing](#required-commands-before-committing)
  - [CI Requirements](#ci-requirements)
- [License](#license)

### Prerequisites

1. [Rust installation page](https://www.rust-lang.org/tools/install)
2. [blackbox_decode](https://github.com/betaflight/blackbox-tools) your BBL to CSV (`--save-headers`, `--index <num>`, and `--limits` parameters may be useful)

### Build

```shell
cargo build --release
```

### Usage
```shell
Usage: ./BlackBox_CSV_Render <input_file1.csv> [<input_file2.csv> ...] [--dps <value>] [--output-dir <directory>] [--butterworth] [--debug] [--step] [--motor] [--setpoint]
  <input_fileX.csv>: Path to one or more input CSV log files (required).
  --dps <value>: Optional. Enables detailed step response plots with the specified
                 deg/s threshold value. Must be a positive number.
                 If --dps is omitted, a general step-response is shown.
  --output-dir <directory>: Optional. Specifies the output directory for generated plots.
                         If omitted, plots are saved in the source folder (input directory).
  --butterworth: Optional. Show Butterworth per-stage PT1 cutoffs for PT2/PT3/PT4 filters
                 as gray curves/lines on gyro and D-term spectrum plots.
  --debug: Optional. Shows detailed metadata information during processing.
  --step: Optional. Generate only step response plots, skipping all other graphs.
  --motor: Optional. Generate only motor spectrum plots, skipping all other graphs.
  --setpoint: Optional. Generate only setpoint-related plots (PIDsum, Setpoint vs Gyro, Setpoint Derivative).
  -h, --help: Show this help message and exit.
  -V, --version: Show version information and exit.

Note: --step, --motor, and --setpoint are non-mutually exclusive and can be combined
(e.g., --step --setpoint generates both step response and setpoint plots).

Arguments can be in any order. Wildcards (e.g., *.csv) are supported by the shell.
```
### Example execution commands
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv
```
```shell
./target/release/BlackBox_CSV_Render path/to/*LOG*.csv --dps 500 --butterworth
```
```shell
./target/release/BlackBox_CSV_Render path1/to/BTFL_*.csv path2/to/EMUF_*.csv --output-dir ./plots --butterworth
```
```shell
./target/release/BlackBox_CSV_Render path/to/ --step --output-dir ./step-only
```
```shell
./target/release/BlackBox_CSV_Render path/to/ --setpoint --output-dir ./setpoint-only
```
```shell
./target/release/BlackBox_CSV_Render path/to/ --step --setpoint --motor --output-dir ./all-selective
```

### Output

#### PNG Files Generated
- `*_Step_Response_stacked_plot_*.png` — Step response analysis with P:D recommendations
- `*_PIDsum_PIDerror_Setpoint_stacked.png` — PIDsum, PID error, and setpoint traces
- `*_SetpointVsGyro_stacked.png` — Setpoint vs. filtered gyro comparison
- `*_GyroVsUnfilt_stacked.png` — Filtered vs. unfiltered gyro comparison with delay estimates
- `*_Gyro_Spectrums_comparative.png` — Frequency-domain gyro amplitude spectrums
- `*_Gyro_PSD_comparative.png` — Gyro power spectral density (dB scale)
- `*_D_Term_Spectrums_comparative.png` — Frequency-domain D-term amplitude spectrums
- `*_D_Term_PSD_comparative.png` — D-term power spectral density (dB scale)
- `*_D_Term_Heatmap_comparative.png` — D-term throttle/frequency heatmap
- `*_Gyro_PSD_Spectrogram_comparative.png` — Gyro spectrogram (PSD vs. time)
- `*_Throttle_Freq_Heatmap_comparative.png` — Throttle/frequency heatmap analysis
- `*_Motor_Spectrums_stacked.png` — Motor output frequency analysis (supports any motor count; colors wrap every 8 motors)

#### Console Output:
- Current P:D ratio and peak analysis with response assessment
- Conservative and Moderate tuning recommendations (with D/D-Min/D-Max values)
- Warning indicators for severe overshoot or unreasonable ratios
- Gyro filtering delay estimates (filtered vs. unfiltered, with confidence)
- Filter configuration parsing and spectrum peak detection summaries
- Use `--debug` flag for additional metadata: header information, flight data key mapping, sample header values, and debug mode identification

#### Code and Output Overview

For a detailed explanation of the program's functionality, especially the step-response calculation and comparison with other tools like PIDtoolbox (Matlab) and PlasmaTree PID-Analyzer (Python), please see [OVERVIEW.md](OVERVIEW.md).

## Development

### Setting Up Development Environment

To set up your development environment with proper formatting and pre-commit hooks:

```bash
# Clone and setup
git clone https://github.com/nerdCopter/BlackBox_CSV_Render.git
cd BlackBox_CSV_Render

# Run setup script (optional but recommended)
chmod +x .github/setup-dev.sh
./.github/setup-dev.sh
```

### Required Commands Before Committing

**⚠️ IMPORTANT**: Always run these commands before committing to avoid CI failures:

```bash
# 1. Check for clippy warnings (must be fixed first)
cargo clippy --all-targets --all-features -- -D warnings

# 2. Format code
cargo fmt --all

# 3. Check formatting compliance
cargo fmt --all -- --check

# 4. Run all tests
cargo test --verbose

# 5. Build release
cargo build --release
```

**The development setup includes an automated pre-commit hook that will:**
- Automatically format your code with `cargo fmt`
- Run clippy checks to catch code issues
- Prevent commits with formatting issues

### CI Requirements

The project enforces strict formatting and code quality standards.

## License

This project is dual-licensed under the terms of the AGPL-3.0-or-later and a commercial license.

- **AGPL-3.0-or-later:** You may use, distribute, and modify this software under the terms of the GNU Affero General Public License, version 3 or any later version. The full license text is available in the [LICENSE](LICENSE) file.

- **Commercial License:** If you wish to use this software in a commercial product without being bound by the terms of the AGPL, you must purchase a commercial license. For more information, please see the [LICENSE_COMMERCIAL](LICENSE_COMMERCIAL) file.
