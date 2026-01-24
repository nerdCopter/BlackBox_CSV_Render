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
Usage: ./BlackBox_CSV_Render <input1> [<input2> ...] [-O|--output-dir <directory>] [--bode] [--butterworth] [--debug] [--dps <value>] [--estimate-optimal-p] [--prop-size <size>] [--prop-pitch <pitch>] [--motor-size <size>] [--motor-kv <kv>] [--lipo <cells>] [--motor-diagonal <mm>] [--motor-width <mm>] [--weight <grams>] [--motor] [--pid] [-R|--recursive] [--setpoint] [--step]
  <inputX>: One or more input CSV files, directories, or shell-expanded wildcards (required).
            Can mix files and directories in a single command.
            - Individual CSV file: path/to/file.csv
            - Directory: path/to/dir/ (finds CSV files only in that directory)
            - Wildcards: *.csv, *LOG*.csv (shell-expanded; works with mixed file and directory patterns)
            Note: Header files (.header.csv, .headers.csv) are automatically excluded.
  -O, --output-dir <directory>: Optional. Specifies the output directory for generated plots.
                              If omitted, plots are saved in the source folder (input directory).
  --bode: Optional. Generate Bode plot analysis (magnitude, phase, coherence).
          NOTE: Requires controlled test flights with system-identification inputs
          (chirp/PRBS). Not recommended for normal flight logs.
  --butterworth: Optional. Show Butterworth per-stage PT1 cutoffs for PT2/PT3/PT4 filters
                 as gray curves/lines on gyro and D-term spectrum plots.
  --debug: Optional. Shows detailed metadata information during processing.
  --dps <value>: Optional. Enables detailed step response plots with the specified
                 deg/s threshold value. Must be a positive number.
                 If --dps is omitted, a general step-response is shown.
  --estimate-optimal-p: Optional. Enable optimal P estimation with physics-aware recommendations.
                        Analyzes response time vs. prop-size targets and noise levels.
  --prop-size <size>: Optional. Specify propeller diameter in inches for optimal P estimation.
                      Valid range: 1.0-15.0 (decimals allowed, e.g., 5.1 or 5.5)
                      Defaults to 5.0 if --estimate-optimal-p is used without this flag.
                      Example: 6-inch frame with 5-inch props → use --prop-size 5
                      Example: 6-inch frame with 5.5-inch props → use --prop-size 5.5
                      Note: This flag is only applied when --estimate-optimal-p is enabled.
                      If --prop-size is provided without --estimate-optimal-p, a warning
                      will be shown and the prop size setting will be ignored.
  --prop-pitch <pitch>: Optional. Specify propeller pitch in inches (e.g., 3.7 for 3.7" pitch).
                        Valid range: 1.0-10.0. Used with --estimate-optimal-p to account for
                        aerodynamic loading differences. Low pitch props have faster response,
                        high pitch props have slower response. Defaults to 4.5" if not specified.
  --motor-size <size>: Optional. Motor stator size (e.g., 2207 for 22mm diameter, 07mm height).
  --motor-kv <kv>: Optional. Motor KV rating (RPM per volt).
  --lipo <cells>: Optional. Battery cell count (e.g., 4S, 5S, 6S).
  --motor-diagonal <mm>: Optional. Frame motor-to-motor diagonal distance in mm.
  --motor-width <mm>: Optional. Frame motor-to-motor width distance in mm.
  --weight <grams>: Optional. Total aircraft weight in grams.
                    Note: All physics parameters (motor, weight, dimensions) are only used
                    when --estimate-optimal-p is enabled. Warnings will be shown if provided
                    without optimal P estimation enabled.
  --motor: Optional. Generate only motor spectrum plots, skipping all other graphs.
  --pid: Optional. Generate only P, I, D activity stacked plot (showing all three PID terms over time).
  -R, --recursive: Optional. When processing directories, recursively find CSV files in subdirectories.
  --setpoint: Optional. Generate only setpoint-related plots (PIDsum, Setpoint vs Gyro, Setpoint Derivative).
  --step: Optional. Generate only step response plots, skipping all other graphs.
  -h, --help: Show this help message and exit.
  -V, --version: Show version information and exit.

Note: --step, --motor, --setpoint, --bode, and --pid are non-mutually exclusive and can be combined
(e.g., --step --setpoint --pid generates step response, setpoint, and PID activity plots).

Arguments can be in any order. Wildcards (e.g., *.csv) are shell-expanded and work with mixed file/directory patterns.
```
### Example execution commands
```shell
# Basic analysis
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv

# With detailed step response and filter display
./target/release/BlackBox_CSV_Render path/to/*LOG*.csv --dps 500 --butterworth

# Basic optimal P estimation
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv --step --estimate-optimal-p --prop-size 5

# Optimal P with pitch parameter
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv --estimate-optimal-p --prop-size 5.1 --prop-pitch 4.0

# Multiple files with output directory
./target/release/BlackBox_CSV_Render path1/*.csv path2/*.csv --output-dir ./plots

# Recursive directory processing
./target/release/BlackBox_CSV_Render path/to/ -R --step --output-dir ./step-only

# Selective plot generation
./target/release/BlackBox_CSV_Render path/to/ --step --setpoint --motor --output-dir ./selective
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
- `*_PID_Activity_stacked.png` — P, I, D term activity over time (stacked plot showing all three PID components)

#### Console Output:
- Current P:D ratio and peak analysis with response assessment
- Conservative and Moderate tuning recommendations (with D/D-Min/D-Max values)
- Warning indicators for severe overshoot or unreasonable ratios
- Optimal P estimation (when --estimate-optimal-p is used):
  - Prop-size-aware Td (time to 50%) analysis
  - Response consistency metrics (CV, std dev)
  - Physics-based P gain recommendations
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
