# Benchmarks

Run from the repository root:

```bash
uv run python benchmarks/bench_flash_screening.py --suite all
```

Useful variants:

```bash
uv run python benchmarks/bench_flash_screening.py --suite length --dtypes float32,bfloat16
uv run python benchmarks/bench_flash_screening.py --suite window --seq-len 2048 --dtype float32
uv run python benchmarks/bench_flash_screening.py --suite length --skip-backward
```

The benchmark uses CUDA events after warmup and reports median latency across
several samples. Inputs are row-normalized to match the screening assumption
that query/key/value vectors are unit length before the screening unit.

## Profiler

Run a training-step profile for the Triton path:

```bash
uv run python benchmarks/profile_flash_screening.py --mode flash --seq-len 2048 --window 128 --dtype bfloat16
```

Compare eager and Triton:

```bash
uv run python benchmarks/profile_flash_screening.py --mode both --seq-len 1024 --window 128 --dtype float32
```

Export Chrome traces for TensorBoard or `chrome://tracing`:

```bash
uv run python benchmarks/profile_flash_screening.py --mode flash --trace-dir traces
```
