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

> **Notice:** Default output is **core plots only**. Use `--extended` for all plots.

```shell
Usage: ./BlackBox_CSV_Render <input1> [<input2> ...] [OPTIONS]

=== INPUT/OUTPUT OPTIONS ===

  <inputX>: CSV files, directories, or wildcards (*.csv). Header files auto-excluded.
  -O, --output-dir <directory>: Output directory (default: source folder).
  -R, --recursive: Recursively find CSV files in subdirectories.

=== PLOT TYPE SELECTION ===

  Default (no flag): Core plots — Step Response, Gyro Spectrums, D-term Spectrums,
                     Setpoint vs Gyro, Gyro vs Unfiltered, Motor Spectrums.

  --extended:      All plots except Bode — adds PIDsum/Error, PID Activity,
                   Setpoint Derivative, Gyro PSD, D-term PSD, and heatmaps
                   (Gyro Spectrogram, Throttle/Freq, D-term) to the core set.

  --step:          Step response plots only.
  --bode:          Bode plot only (requires chirp/sweep system-id test flight).

  --extended and --step/--bode are combinable: --extended --bode adds Bode to
  the full set; --step --bode generates both in isolation.

=== ANALYSIS OPTIONS ===

  --butterworth: Show Butterworth PT1 cutoffs on gyro/D-term spectrum plots.
  --dps <value>: Deg/s threshold for detailed step response plots (positive number).
  --estimate-optimal-p: [EXPERIMENTAL] Optimal P estimation from throttle-punch
                        dynamics. Requires .headers.csv; skips if absent.

=== GENERAL ===

  --debug: Show detailed metadata during processing.
  -h, --help: Show this help message and exit.
  -V, --version: Show version information.
```

Arguments can be in any order. Wildcards (e.g., *.csv) are shell-expanded and work with mixed file/directory patterns.

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
./target/release/BlackBox_CSV_Render path/to/ -R --step --output-dir ./step-only
```
```shell
./target/release/BlackBox_CSV_Render path/to/ --extended --output-dir ./all-plots
```
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv --step --estimate-optimal-p
```

### Output

#### PNG Files Generated

**Core (default):**
- `*_Step_Response_stacked_plot_*.png` — Step response analysis with P:D recommendations
- `*_SetpointVsGyro_stacked.png` — Setpoint vs. filtered gyro comparison
- `*_GyroVsUnfilt_stacked.png` — Filtered vs. unfiltered gyro comparison with delay estimates
- `*_Gyro_Spectrums_comparative.png` — Frequency-domain gyro amplitude spectrums
- `*_D_Term_Spectrums_comparative.png` — Frequency-domain D-term amplitude spectrums
- `*_Motor_Spectrums_stacked.png` — Motor output frequency analysis (supports any motor count; colors wrap every 8 motors)

**Extended (`--extended` adds these to the core set):**
- `*_PIDsum_PIDerror_Setpoint_stacked.png` — PIDsum, PID error, and setpoint traces
- `*_PID_Activity_stacked.png` — P, I, D term activity over time
- `*_SetpointDerivative_stacked.png` — Setpoint rate-of-change / feed-forward proxy
- `*_Gyro_PSD_comparative.png` — Gyro power spectral density (dB scale)
- `*_D_Term_PSD_comparative.png` — D-term power spectral density (dB scale)
- `*_D_Term_Heatmap_comparative.png` — D-term throttle/frequency heatmap
- `*_Gyro_PSD_Spectrogram_comparative.png` — Gyro spectrogram (PSD vs. time)
- `*_Throttle_Freq_Heatmap_comparative.png` — Throttle/frequency heatmap analysis

#### Console Output:
- Current P:D ratio and peak analysis with response assessment
- Conservative and Moderate tuning recommendations (with D/D-Min/D-Max values)
- Warning indicators for severe overshoot or unreasonable ratios
- Gyro filtering delay estimates (filtered vs. unfiltered, with confidence)
- Filter configuration parsing and spectrum peak detection summaries
- Optimal P estimation (`--estimate-optimal-p`): Td timing, target deviation, noise level, consistency, P/D recommendations and skip-reason warnings
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
