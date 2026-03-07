Fetch mnist dataset and convert it to parquet.

Usage:
```
uv install
uv run get_dataset.py
```

Then move to the root directory and run

```
cargo run --example mnist_to_parquet data/mnist data/
```
