## Experiment with A.I to create a RUST program to read Betaflight Blackbox CSV and produce meaningful graphs.

### Prerequisites

1. https://www.rust-lang.org/tools/install
2. [blackbox_decode](https://github.com/betaflight/blackbox-tools) your BBL to CSV

### Build and execute

```shell
cargo build --release && ./target/release/RUST_BlackBox_CSV_Render path/to/BTFL_Log.csv
ls -lhrt *.png
```
