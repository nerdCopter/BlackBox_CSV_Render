## This is an experimental RUST program, mostly created via A.I., to read Betaflight/EmuFlight Blackbox BBL or CSV logs and produce meaningful graphs.

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
2. Optional: `.BBL`/`.BFL` files are read directly. Multi-flight files can produce one report per eligible
   flight; low-value flights are skipped unless `-F`/`--force-export` is used. Manual
   [blackbox_decode](https://github.com/betaflight/blackbox-tools) preprocessing to CSV is only needed for its
   selective-extraction flags (`--save-headers`, `--index <num>`, `--limits`).

### Build

```shell
cargo build --release
```

### Usage

> **Notice:** Default output is **core plots only**. Use `--extended` for all plots.

```shell
Usage: ./BlackBox_CSV_Render <input1> [<input2> ...] [OPTIONS]

=== INPUT/OUTPUT OPTIONS ===

  <inputX>: CSV, BBL, or BFL files, directories, or wildcards (*.csv, *.bbl, *.bfl). Header
            files auto-excluded. A multi-flight .bbl/.bfl can produce one report per eligible flight.
  -O, --output-dir <directory>: Output directory (default: source folder).
  -R, --recursive: Recursively find CSV/BBL/BFL files in subdirectories.
  -F, --force-export: Export .bbl flights bbl_parser would otherwise skip as low-value
                       (very short / low data density / minimal gyro activity).
  -K, --keep: Keep the exported .csv/.headers.csv files (source folder, or -O/--output-dir)

=== PLOT TYPE SELECTION ===

  --core           [default] Step Response, Gyro Spectrums, D-term Spectrums,
                   Setpoint vs Gyro, Gyro vs Unfiltered, Motor Spectrums,
                   RC Command Activity.
  --extended       All plots except Bode — adds PIDsum/Error, PID Activity,
                   Setpoint Derivative, Gyro PSD, D-term PSD, heatmaps, and
                   Motor vs eRPM (requires bidirectional DShot telemetry).
  --step           Step response only.
  --bode           Bode only (requires chirp/sweep system-id test flight).
  --desync         Motor vs eRPM plot only (needs eRPM telemetry). Desync
                   detection itself (Fallback tier) still runs without it.

=== ANALYSIS OPTIONS ===

  --butterworth    Show Butterworth PT1 cutoffs on gyro/D-term spectrum plots.
  --dps <value>    Deg/s threshold for detailed step response plots (positive number).
  --estimate-optimal-p  [EXPERIMENTAL] Optimal P estimation from throttle-punch
                        dynamics. Requires .headers.csv; skips if absent.
                        Its Td target always profiles the full file, ignoring
                        --start/--end; only its Td measurement is trimmed.

=== TIME WINDOW ===

  --start <seconds>  Trim analysis to this offset onward, relative to the
                     start of the log. Omit to start at the log start.
  --end <seconds>    Trim analysis up to this offset, relative to the
                     start of the log. Omit to end at the log end.
                     Independent — use either or both. Applies before every
                     analysis and plot (step response may skip if the
                     trimmed window is too short).

=== GENERAL ===

  --debug          Show detailed metadata during processing.
  -h, --help       Show this help message and exit.
  -V, --version    Show version information.
```

Arguments can be in any order. Wildcards (e.g., *.csv) are shell-expanded and work with mixed file/directory patterns.

### Example execution commands
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv
```
```shell
./target/release/BlackBox_CSV_Render path/to/EMUF_Log.BBL
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
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv --start 40 --desync
```

### Time-Window Trim: How To Pick `--start`/`--end`

`--start`/`--end` are seconds relative to the start of the log (`0s` = log start). Trimming drops
every row outside the window before any plot or analysis runs, so it affects every Phase 2
analysis and plot — except `--estimate-optimal-p`'s Phase 1 aircraft profiling, which always
re-reads the full file regardless of `--start`/`--end`, and Motor Desync Detection's baseline
statistics, which are always computed from the full log even when a window is applied (see both
caveats below).

**General workflow:**
1. Run the full log first (no `--start`/`--end`). A run prints `Note: Log spans absolute time
   Xs-Ys (duration Ds)` whenever the log has a positive duration (both a first and last
   timestamp, with the last one later) — that line is informational only, for cross-referencing
   against OSD/video overlays that also show raw flight-controller uptime. A one-row log, or one
   where every row shares the same timestamp, has no duration and omits this line —
   `--start`/`--end` aren't meaningful there either. Every plot, and every report table timestamp
   (e.g. the `Possible`/`De Facto`/`Fallback` "Times (s)" column in the Motor Desync Detection
   report), reads on the same `0s`-at-log-start timeline as `--start`/`--end` — pass the event's
   timestamp straight to `--start`/`--end` with margin, no subtraction needed (e.g. event at
   `9.85s` → `--start` around `9.0` or earlier for margin).
2. Trim with real margin *before* that timestamp — don't cut the window right up against the
   event. Re-run and compare the plot to the full-log version.
3. If a report table's flag depends on a statistic computed only from the trimmed window itself
   (e.g. an axis's percentile-based classification), confirm the flag still appears after
   trimming. If it disappeared, move `--start` earlier (more margin) and re-check — the boundary
   between "flag holds" and "flag disappears" isn't a fixed number of seconds and can flip on trim
   values close together. Motor Desync Detection is exempt from this: see below.

**Motor Desync Detection specifically:** fixed in IT #182. The `Possible`/`De Facto`/`Fallback`
tiers (`src/plot_functions/motor_desync.rs`) always score each motor against statistics built from
the **full, untrimmed log** — its own high-command runs elsewhere in the flight, their eRPM
baseline, and (for `Possible`) a 90th-percentile eRPM reference — regardless of `--start`/`--end`.
Only which flagged events get *reported* is restricted to the trimmed window. A tight trim can
therefore make the `Motor_vs_eRPM` plot readable without losing the table flag, unlike before the
fix. `--estimate-optimal-p` in particular still splits across trim and no-trim — see the Phase
1/Phase 2 note above and `OVERVIEW.md`.

### Output

#### PNG Files Generated

**Core (default):**
- `*_Step_Response_stacked_plot_*.png` — Step response analysis with P:D recommendations
- `*_SetpointVsGyro_stacked.png` — Setpoint vs. filtered gyro comparison
- `*_GyroVsUnfilt_stacked.png` — Filtered vs. unfiltered gyro comparison with delay estimates
- `*_Gyro_Spectrums_comparative.png` — Frequency-domain gyro amplitude spectrums
- `*_D_Term_Spectrums_comparative.png` — Frequency-domain D-term amplitude spectrums
- `*_Motor_Spectrums_stacked.png` — Motor output frequency analysis (supports any motor count; colors wrap every 8 motors)
- `*_RC_Command_Activity_stacked.png` — Setpoint vs. RC Command overlay per axis; visualizes blocky/unfiltered stick input against the flight controller's response

**Extended (`--extended` adds these to the core set):**
- `*_PIDsum_PIDerror_Setpoint_stacked.png` — PIDsum, PID error, and setpoint traces
- `*_PID_Activity_stacked.png` — P, I, D term activity over time
- `*_SetpointDerivative_stacked.png` — Setpoint rate-of-change / feed-forward proxy
- `*_Gyro_PSD_comparative.png` — Gyro power spectral density (dB scale)
- `*_D_Term_PSD_comparative.png` — D-term power spectral density (dB scale)
- `*_D_Term_Heatmap_comparative.png` — D-term throttle/frequency heatmap
- `*_Gyro_PSD_Spectrogram_comparative.png` — Gyro spectrogram (PSD vs. time)
- `*_Throttle_Freq_Heatmap_comparative.png` — Throttle/frequency heatmap analysis
- `*_Motor_vs_eRPM_stacked.png` — One row per motor, commanded output and eRPM telemetry overlaid, each normalized to its own 0-100% range; vertical markers at every Motor Desync Detection event timestamp. Skipped when the log has no eRPM telemetry.

#### Markdown Report (always generated)

- `*_report.md` — Structured flight report written alongside PNGs on every run. Sections: Metadata (firmware, PIDs, sample rate, trimmed time window when `--start`/`--end` was used, gyroUnfilt source), Filter Configuration (LPF1/LPF2/IMUF/Pseudo-Kalman table, Dynamic Notch, RPM filter), PID Tuning, Step Response Analysis (Roll/Pitch with P:D assessment and setpoint authority), Gyro Analysis (filtering delay, confidence, spectrum peaks per axis), D-Term Analysis (filtering delay with N/A reason, spectrum peaks), Motor Oscillation (per-motor sliding-window spectrum check, catches a brief burst a whole-log average would dilute away), Motor Desync Detection (per-motor motor[N] vs eRPM[N] divergence, self-relative to that same motor's own behavior elsewhere in the flight; De Facto/Possible confidence tiers; requires bidirectional DShot telemetry), Stick Input Smoothness (RC Command step detection, with an rc_smoothing recommendation when an axis is classified Blocky), Stick Position & Rate Analysis (computed regardless of plot selection, not gated by `--core`/`--extended`/etc. — per-axis Peak Stick and Center/Mid/High/Saturation zone-time percentages relative to true full-stick deflection (not this flight's own peak), Saturation Events, Center-zone reversal rate, P95 Setpoint/P95 Gyro Achieved, and Configured Max Rate/Rate Headroom from the header's rate-curve config; omitted from the report only when the log has no RC Command samples with a positive peak for any axis), links to all generated PNGs, and a Skipped Plots list naming any enabled plot type with no plottable data for any axis, unless a stale PNG from an earlier run causes it to be classified as generated instead. Optimal P Estimation and Bode Analysis sections appear when those features are active.

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
