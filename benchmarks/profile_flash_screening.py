from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile, record_function

from flash_screening import flash_screening
from flash_screening.eager import screening as eager_screening


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


def parse_dtype(value: str) -> torch.dtype:
    try:
        return DTYPES[value]
    except KeyError as exc:
        raise argparse.ArgumentTypeError(
            f"unknown dtype: {value}; choose from {', '.join(DTYPES)}"
        ) from exc


def make_inputs(args: argparse.Namespace) -> tuple[torch.Tensor, ...]:
    shape_qk = (args.batch, args.heads, args.seq_len, args.key_dim)
    shape_v = (args.batch, args.heads, args.seq_len, args.value_dim)
    query = F.normalize(torch.randn(shape_qk, device="cuda"), dim=-1).to(args.dtype)
    key = F.normalize(torch.randn(shape_qk, device="cuda"), dim=-1).to(args.dtype)
    value = F.normalize(torch.randn(shape_v, device="cuda"), dim=-1).to(args.dtype)
    acceptance = torch.full((args.heads,), 0.9, device="cuda", dtype=args.dtype)
    window = torch.full((args.heads,), float(args.window), device="cuda", dtype=args.dtype)
    return (
        query.detach().clone().requires_grad_(),
        key.detach().clone().requires_grad_(),
        value.detach().clone().requires_grad_(),
        acceptance.detach().clone().requires_grad_(),
        window.detach().clone().requires_grad_(),
    )


def clear_grads(tensors: tuple[torch.Tensor, ...]) -> None:
    for tensor in tensors:
        tensor.grad = None


def flash_step(tensors: tuple[torch.Tensor, ...], grad_output: torch.Tensor) -> None:
    query, key, value, acceptance, window = tensors
    output = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )
    output.backward(grad_output)


def eager_step(tensors: tuple[torch.Tensor, ...], grad_output: torch.Tensor) -> None:
    query, key, value, acceptance, window = tensors
    output = eager_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )
    assert isinstance(output, torch.Tensor)
    output.backward(grad_output)


def run_step(
    name: str,
    step_fn: Callable[[tuple[torch.Tensor, ...], torch.Tensor], None],
    tensors: tuple[torch.Tensor, ...],
    grad_output: torch.Tensor,
) -> None:
    clear_grads(tensors)
    with record_function(f"{name}_screening_step"):
        step_fn(tensors, grad_output)


def profile_mode(
    *,
    name: str,
    args: argparse.Namespace,
    step_fn: Callable[[tuple[torch.Tensor, ...], torch.Tensor], None],
) -> None:
    tensors = make_inputs(args)
    grad_output = torch.randn_like(tensors[2])

    for _ in range(args.warmup):
        run_step(name, step_fn, tensors, grad_output)
    torch.cuda.synchronize()

    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        acc_events=True,
    ) as prof:
        for _ in range(args.steps):
            run_step(name, step_fn, tensors, grad_output)
            prof.step()
    torch.cuda.synchronize()

    print(f"== {name} ==")
    print(
        prof.key_averages().table(
            sort_by=args.sort_by,
            row_limit=args.row_limit,
        )
    )

    if args.trace_dir is not None:
        args.trace_dir.mkdir(parents=True, exist_ok=True)
        trace_path = args.trace_dir / f"{name}_screening_trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"trace: {trace_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile eager vs Triton screening")
    parser.add_argument("--mode", choices=["flash", "eager", "both"], default="flash")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--key-dim", type=int, default=64)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--window", type=int, default=128)
    parser.add_argument("--dtype", type=parse_dtype, default=torch.bfloat16)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--row-limit", type=int, default=25)
    parser.add_argument("--sort-by", default="self_cuda_time_total")
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--trace-dir", type=Path)
    return parser


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this profiler")

    parser = build_parser()
    args = parser.parse_args()
    torch.manual_seed(0)

    print(f"device: {torch.cuda.get_device_name()}")
    print(f"torch: {torch.__version__}")
    print(
        "shape: "
        f"B={args.batch}, H={args.heads}, L={args.seq_len}, "
        f"Dk={args.key_dim}, Dv={args.value_dim}, W={args.window}, "
        f"dtype={str(args.dtype).replace('torch.', '')}"
    )
    print()

    if args.mode in ("flash", "both"):
        profile_mode(name="flash", args=args, step_fn=flash_step)
    if args.mode in ("eager", "both"):
        profile_mode(name="eager", args=args, step_fn=eager_step)


if __name__ == "__main__":
    main()
