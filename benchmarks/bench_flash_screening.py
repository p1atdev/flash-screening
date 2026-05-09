from __future__ import annotations

import argparse
import contextlib
import statistics
from collections.abc import Callable, Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from flash_screening import flash_screening
from flash_screening.eager import screening as eager_screening


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True)
class BenchConfig:
    batch: int
    heads: int
    seq_len: int
    key_dim: int
    value_dim: int
    window: int
    dtype: torch.dtype
    device: str


@dataclass(frozen=True)
class TimingConfig:
    samples: int
    warmup: int
    iters: int


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_csv_dtypes(value: str) -> list[torch.dtype]:
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in DTYPES]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown dtype(s): {', '.join(unknown)}; choose from {', '.join(DTYPES)}"
        )
    return [DTYPES[name] for name in names]


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def make_inputs(config: BenchConfig) -> tuple[torch.Tensor, ...]:
    shape_qk = (
        config.batch,
        config.heads,
        config.seq_len,
        config.key_dim,
    )
    shape_v = (
        config.batch,
        config.heads,
        config.seq_len,
        config.value_dim,
    )
    query = F.normalize(torch.randn(shape_qk, device=config.device), dim=-1).to(
        config.dtype
    )
    key = F.normalize(torch.randn(shape_qk, device=config.device), dim=-1).to(
        config.dtype
    )
    value = F.normalize(torch.randn(shape_v, device=config.device), dim=-1).to(
        config.dtype
    )
    acceptance = torch.full(
        (config.heads,),
        0.9,
        device=config.device,
        dtype=config.dtype,
    )
    window = torch.full(
        (config.heads,),
        float(config.window),
        device=config.device,
        dtype=config.dtype,
    )
    return query, key, value, acceptance, window


def cuda_time_ms(fn: Callable[[], object], timing: TimingConfig) -> float:
    for _ in range(timing.warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(timing.samples):
        start.record()
        for _ in range(timing.iters):
            fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) / timing.iters)
    return statistics.median(samples)


def flash_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    return flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )


def eager_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
) -> torch.Tensor:
    output = eager_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )
    assert isinstance(output, torch.Tensor)
    return output


def bench_forward(config: BenchConfig, timing: TimingConfig) -> tuple[float, float]:
    query, key, value, acceptance, window = make_inputs(config)
    with torch.no_grad():
        flash_forward(query, key, value, acceptance, window)
        eager_forward(query, key, value, acceptance, window)
        flash_ms = cuda_time_ms(
            lambda: flash_forward(query, key, value, acceptance, window),
            timing,
        )
        eager_ms = cuda_time_ms(
            lambda: eager_forward(query, key, value, acceptance, window),
            timing,
        )
    return eager_ms, flash_ms


def make_trainable_inputs(config: BenchConfig) -> tuple[torch.Tensor, ...]:
    query, key, value, acceptance, window = make_inputs(config)
    return (
        query.detach().clone().requires_grad_(),
        key.detach().clone().requires_grad_(),
        value.detach().clone().requires_grad_(),
        acceptance.detach().clone().requires_grad_(),
        window.detach().clone().requires_grad_(),
    )


def clear_grads(tensors: Iterable[torch.Tensor]) -> None:
    for tensor in tensors:
        tensor.grad = None


def bench_backward(config: BenchConfig, timing: TimingConfig) -> tuple[float, float]:
    query, key, value, acceptance, window = make_trainable_inputs(config)
    grad_output = torch.randn_like(value)

    def run(fn: Callable[..., torch.Tensor]) -> None:
        clear_grads((query, key, value, acceptance, window))
        output = fn(query, key, value, acceptance, window)
        output.backward(grad_output)

    run(flash_forward)
    run(eager_forward)
    torch.cuda.synchronize()

    flash_ms = cuda_time_ms(lambda: run(flash_forward), timing)
    eager_ms = cuda_time_ms(lambda: run(eager_forward), timing)
    return eager_ms, flash_ms


def print_header(config: BenchConfig, suite: str) -> None:
    print(f"suite: {suite}")
    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}")
    print(
        "base: "
        f"B={config.batch}, H={config.heads}, "
        f"Dk={config.key_dim}, Dv={config.value_dim}"
    )
    print()


def print_row(kind: str, config: BenchConfig, eager_ms: float, flash_ms: float) -> None:
    print(
        f"{kind:8s} {dtype_name(config.dtype):8s} "
        f"L={config.seq_len:<5d} W={config.window:<5d} "
        f"eager={eager_ms:9.3f} ms  "
        f"flash={flash_ms:9.3f} ms  "
        f"speedup={eager_ms / flash_ms:8.2f}x"
    )


def run_case(
    *,
    config: BenchConfig,
    forward_timing: TimingConfig,
    backward_timing: TimingConfig,
    skip_forward: bool,
    skip_backward: bool,
) -> None:
    if not skip_forward:
        with contextlib.suppress(torch.cuda.OutOfMemoryError):
            eager_ms, flash_ms = bench_forward(config, forward_timing)
            print_row("forward", config, eager_ms, flash_ms)

    if not skip_backward:
        with contextlib.suppress(torch.cuda.OutOfMemoryError):
            eager_ms, flash_ms = bench_backward(config, backward_timing)
            print_row("fwd+bwd", config, eager_ms, flash_ms)


def run_length_suite(args: argparse.Namespace) -> None:
    base = base_config(args)
    print_header(base, "length")
    for dtype in args.dtypes:
        for seq_len in args.seq_lens:
            config = replace_config(base, dtype=dtype, seq_len=seq_len)
            run_case(
                config=config,
                forward_timing=args.forward_timing,
                backward_timing=args.backward_timing,
                skip_forward=args.skip_forward,
                skip_backward=args.skip_backward,
            )
        print()


def run_window_suite(args: argparse.Namespace) -> None:
    base = base_config(args)
    print_header(base, "window")
    for dtype in args.dtypes:
        for window in args.window_sizes:
            config = replace_config(base, dtype=dtype, window=window)
            run_case(
                config=config,
                forward_timing=args.forward_timing,
                backward_timing=args.backward_timing,
                skip_forward=args.skip_forward,
                skip_backward=args.skip_backward,
            )
        print()


def base_config(args: argparse.Namespace) -> BenchConfig:
    return BenchConfig(
        batch=args.batch,
        heads=args.heads,
        seq_len=args.seq_len,
        key_dim=args.key_dim,
        value_dim=args.value_dim,
        window=args.window,
        dtype=args.dtypes[0],
        device="cuda",
    )


def replace_config(
    config: BenchConfig,
    *,
    dtype: torch.dtype | None = None,
    seq_len: int | None = None,
    window: int | None = None,
) -> BenchConfig:
    return BenchConfig(
        batch=config.batch,
        heads=config.heads,
        seq_len=config.seq_len if seq_len is None else seq_len,
        key_dim=config.key_dim,
        value_dim=config.value_dim,
        window=config.window if window is None else window,
        dtype=config.dtype if dtype is None else dtype,
        device=config.device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark eager vs Triton screening")
    parser.add_argument("--suite", choices=["length", "window", "all"], default="all")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument(
        "--seq-lens",
        type=parse_csv_ints,
        default=parse_csv_ints("256,512,1024,2048,4096"),
    )
    parser.add_argument(
        "--window-sizes",
        type=parse_csv_ints,
        default=parse_csv_ints("64,128,256,512,1024,2048"),
    )
    parser.add_argument(
        "--dtypes",
        "--dtype",
        dest="dtypes",
        type=parse_csv_dtypes,
        default=parse_csv_dtypes("float32,bfloat16"),
    )
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--backward-warmup", type=int, default=3)
    parser.add_argument("--backward-iters", type=int, default=10)
    parser.add_argument("--skip-forward", action="store_true")
    parser.add_argument("--skip-backward", action="store_true")
    return parser


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")

    parser = build_parser()
    args = parser.parse_args()
    args.forward_timing = TimingConfig(
        samples=args.samples,
        warmup=args.warmup,
        iters=args.iters,
    )
    args.backward_timing = TimingConfig(
        samples=args.samples,
        warmup=args.backward_warmup,
        iters=args.backward_iters,
    )

    torch.manual_seed(0)
    if args.suite in ("length", "all"):
        run_length_suite(args)
    if args.suite in ("window", "all"):
        run_window_suite(args)


if __name__ == "__main__":
    main()
