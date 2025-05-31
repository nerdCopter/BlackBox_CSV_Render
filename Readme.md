## This is an experimental RUST program, mostly created via A.I., to read Betaflight Blackbox CSV and produce meaningful graphs.

### Prerequisites

1. https://www.rust-lang.org/tools/install
2. [blackbox_decode](https://github.com/betaflight/blackbox-tools) your BBL to CSV (`--save-headers`, `--index <num>`, and `--limits` parameters may be useful)

### Build

```shell
cargo build --release
```

### Example execution commands
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv
```
```shell
./target/release/BlackBox_CSV_Render path/to/BTFL_Log.csv [--dps [<value>]]
```
```shell
./target/release/BlackBox_CSV_Render path/to/*.csv --dps
```
```shell
./target/release/BlackBox_CSV_Render path1/to/BTFL_Log.csv path2/to/*.csv --dps 360
```

### Output
- PNG files are generated in the current directory.

### Licensing still under consideration.
- Some resources used for the AI prompting included the following, but only for inspiration.
- No code was reused as reported by AI interrogation, and therefore do not require their associated licensing.
- https://github.com/KoffeinFlummi/bucksaw/ with GPL v3.
- https://github.com/Plasmatree/PID-Analyzer/ with Apache License 2.0 and "THE BEER-WARE LICENSE" (Revision 42).
- https://github.com/bw1129/PIDtoolbox/ with GPL v3 and "THE BEER-WARE LICENSE" (Revision 42).
