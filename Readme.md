## This is an experimental RUST program, mostly created via A.I., to read Betaflight Blackbox CSV and produce meaningful graphs.

### Prerequisites

1. https://www.rust-lang.org/tools/install
2. [blackbox_decode](https://github.com/betaflight/blackbox-tools) your BBL to CSV

### Build and execute

```shell
cargo build --release
./target/release/RUST_BlackBox_CSV_Render path/to/BTFL_Log.csv
ls *.png
```

### Licensing still under consideration.
- Some resources used for the AI prompting included the following, but only for inspiration.
- No code was reused as reported by AI interrogation, and therefore do not require their associated licensing.
- https://github.com/KoffeinFlummi/bucksaw/ with GPL v3.
- https://github.com/Plasmatree/PID-Analyzer/ with Apache License 2.0 and "THE BEER-WARE LICENSE" (Revision 42).
- https://github.com/bw1129/PIDtoolbox/ with GPL v3 and "THE BEER-WARE LICENSE" (Revision 42).
