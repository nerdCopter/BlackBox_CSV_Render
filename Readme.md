## This is an experimental RUST program, mostly created via A.I., to read Betaflight Blackbox CSV and produce meaningful graphs.

### Prerequisites

1. https://www.rust-lang.org/tools/install
2. [blackbox_decode](https://github.com/betaflight/blackbox-tools) your BBL to CSV (`--save-headers`, `--index <num>`, and `--limits` parameters may be useful)

### Build

```shell
cargo build --release
```

### Usage
```shell
Usage: ./BlackBox_CSV_Render <input_file1.csv> [<input_file2.csv> ...] [--dps [<value>]] [--out-dir <directory>]
  <input_fileX.csv>: Path to one or more input CSV log files (required).
  --dps [<value>]: Optional. Enables detailed step response plots.
                   If <value> (deg/s threshold) is provided, it's used.
                   If <value> is omitted, defaults to 500.
                   If --dps is omitted, a general step-response is shown.
  --out-dir <directory>: Optional. Specifies the output directory for generated plots.
                         If omitted, plots are saved in the source folder (input file's directory).

Arguments can be in any order. Wildcards (e.g., *.csv) are supported by the shell.
```
### Example execution commands
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv
```
```shell
./target/release/BlackBox_CSV_Render path/to/*LOG*.csv --dps
```
```shell
./target/release/BlackBox_CSV_Render path1/to/BTFL_*.csv path2/to/EMUF_*.csv --dps 360 --out-dir ./plots
```

### Output
- PNG files are generated in the source folder (input file's directory), unless specified by the `--out-dir` parameter.

### Code Overview

For a detailed explanation of the program's functionality, especially the step-response calculation and comparison with other tools like PIDtoolbox (Matlab) and PlasmaTree PID-Analyzer (Python), please see [Overview.md](Overview.md).

### Licensing still under consideration.
- Some resources used for the AI prompting included the following, but only for inspiration.
- No code was reused as reported by AI interrogation, and therefore do not require their associated licensing.
- https://github.com/KoffeinFlummi/bucksaw/ with GPL v3.
- https://github.com/Plasmatree/PID-Analyzer/ with Apache License 2.0 and "THE BEER-WARE LICENSE" (Revision 42).
- https://github.com/bw1129/PIDtoolbox/ with GPL v3 and "THE BEER-WARE LICENSE" (Revision 42).
