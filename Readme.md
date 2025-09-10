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
Usage: ./BlackBox_CSV_Render <input_file1.csv> [<input_file2.csv> ...] [--dps <value>] [--output-dir <directory>]
  <input_fileX.csv>: Path to one or more input CSV log files (required).
  --dps <value>: Optional. Enables detailed step response plots with the specified
                 deg/s threshold value. Must be a positive number.
                 If --dps is omitted, a general step-response is shown.
  --output-dir <directory>: Optional. Specifies the output directory for generated plots.
                         If omitted, plots are saved in the source folder (input file's directory).

Arguments can be in any order. Wildcards (e.g., *.csv) are supported by the shell.
```
### Example execution commands
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv
```
```shell
./target/release/BlackBox_CSV_Render path/to/*LOG*.csv --dps 500
```
```shell
./target/release/BlackBox_CSV_Render path1/to/BTFL_*.csv path2/to/EMUF_*.csv --dps 360 --output-dir ./plots
```

### Output
- PNG files are generated in the source folder (input file's directory), unless specified by the `--output-dir` parameter.

### Code Overview

For a detailed explanation of the program's functionality, especially the step-response calculation and comparison with other tools like PIDtoolbox (Matlab) and PlasmaTree PID-Analyzer (Python), please see [Overview.md](Overview.md).

## Development

### Setting Up Development Environment

To set up your development environment with proper formatting and pre-commit hooks:

```bash
# Clone and setup
git clone <repository-url>
cd BlackBox_CSV_Render

# Run setup script (optional but recommended)
chmod +x .github/setup-dev.sh
./.github/setup-dev.sh
```

### Required Commands Before Committing

**⚠️ IMPORTANT**: Always run these commands before committing to avoid CI failures:

```bash
# 1. Format code (REQUIRED)
cargo fmt --all

# 2. Check formatting compliance
cargo fmt --all -- --check

# 3. Check for clippy warnings (treated as errors)
cargo clippy --all-targets --all-features -- -D warnings

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
