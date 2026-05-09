from __future__ import annotations

from typing import Any, Literal, overload, cast

import torch
import triton
import triton.language as tl

from .eager import screening as eager_screening


_MAX_KERNEL_BLOCK_D = 256


@triton.jit
def _flash_screening_forward_kernel(
    query,
    key,
    value,
    acceptance,
    window,
    position_ids,
    attention_mask,
    output,
    score,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    HAS_ATTENTION_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_SCORE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DK: tl.constexpr,
    BLOCK_DV: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // NUM_HEADS
    head_idx = pid_bh - batch_idx * NUM_HEADS
    bh_offset = pid_bh * SEQ_LEN

    query_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    key_offsets = tl.arange(0, BLOCK_N)
    key_dim_offsets = tl.arange(0, BLOCK_DK)
    value_dim_offsets = tl.arange(0, BLOCK_DV)

    q_ptrs = (
        query
        + (bh_offset + query_offsets[:, None]) * KEY_DIM
        + key_dim_offsets[None, :]
    )
    q_mask = (query_offsets[:, None] < SEQ_LEN) & (
        key_dim_offsets[None, :] < KEY_DIM
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    acceptance_h = tl.load(acceptance + head_idx).to(tl.float32)
    inv_acceptance_h = 1.0 / acceptance_h

    if IS_CAUSAL:
        window_h = tl.load(window + head_idx).to(tl.float32)
    else:
        window_h = 1.0

    if IS_CAUSAL and not HAS_POSITION_IDS:
        window_span = tl.minimum(
            window_h,
            tl.full((), SEQ_LEN + BLOCK_M, dtype=tl.float32),
        ).to(tl.int64) + 1
        key_block_start = tl.maximum(0, pid_m * BLOCK_M - window_span)
        key_block_start = (key_block_start // BLOCK_N) * BLOCK_N
        key_block_stop = tl.minimum(SEQ_LEN, (pid_m + 1) * BLOCK_M)
    else:
        key_block_start = 0
        key_block_stop = SEQ_LEN

    start_n = key_block_start
    while start_n < key_block_stop:
        key_indices = start_n + key_offsets
        k_ptrs = (
            key
            + (bh_offset + key_indices[None, :]) * KEY_DIM
            + key_dim_offsets[:, None]
        )
        k_mask = (key_indices[None, :] < SEQ_LEN) & (
            key_dim_offsets[:, None] < KEY_DIM
        )
        k = tl.load(k_ptrs, mask=k_mask, other=0.0).to(tl.float32)

        similarity = tl.dot(q, k, input_precision="tf32")
        relevance = 1.0 - (1.0 - similarity) * inv_acceptance_h
        relevance = tl.maximum(relevance, 0.0)
        relevance = relevance * relevance

        valid_relevance = (query_offsets[:, None] < SEQ_LEN) & (
            key_indices[None, :] < SEQ_LEN
        )

        if IS_CAUSAL:
            if HAS_POSITION_IDS:
                q_pos = tl.load(
                    position_ids + batch_idx * SEQ_LEN + query_offsets,
                    mask=query_offsets < SEQ_LEN,
                    other=0.0,
                ).to(tl.float32)
                k_pos = tl.load(
                    position_ids + batch_idx * SEQ_LEN + key_indices,
                    mask=key_indices < SEQ_LEN,
                    other=0.0,
                ).to(tl.float32)
                position_diff = k_pos[None, :] - q_pos[:, None]
            else:
                position_diff = key_indices[None, :].to(tl.float32) - query_offsets[
                    :, None
                ].to(tl.float32)

            softmask_valid = (-window_h < position_diff) & (position_diff <= 0.0)
            softmask = 0.5 * (tl.cos(3.141592653589793 * position_diff / window_h) + 1.0)
            relevance = tl.where(softmask_valid, relevance * softmask, 0.0)

        relevance = tl.where(valid_relevance, relevance, 0.0)

        if HAS_ATTENTION_MASK:
            key_mask = tl.load(
                attention_mask + batch_idx * SEQ_LEN + key_indices,
                mask=key_indices < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)
            relevance = relevance * key_mask[None, :]

        if RETURN_SCORE:
            score_ptrs = (
                score
                + (bh_offset + query_offsets[:, None]) * SEQ_LEN
                + key_indices[None, :]
            )
            tl.store(score_ptrs, relevance, mask=valid_relevance)

        v_ptrs = (
            value
            + (bh_offset + key_indices[:, None]) * VALUE_DIM
            + value_dim_offsets[None, :]
        )
        v_mask = (key_indices[:, None] < SEQ_LEN) & (
            value_dim_offsets[None, :] < VALUE_DIM
        )
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)
        acc += tl.dot(relevance, v, input_precision="tf32")

        start_n += BLOCK_N

    value_mask = value_dim_offsets < VALUE_DIM
    squared = tl.where(value_mask[None, :], acc * acc, 0.0)
    norm = tl.sqrt(tl.sum(squared, axis=1))
    exp_neg_2_norm = tl.exp(-2.0 * norm)
    tanh_norm = (1.0 - exp_neg_2_norm) / (1.0 + exp_neg_2_norm)
    scale = tl.where(norm > EPS, tanh_norm / tl.maximum(norm, EPS), 1.0)
    output_values = acc * scale[:, None]

    out_ptrs = (
        output
        + (bh_offset + query_offsets[:, None]) * VALUE_DIM
        + value_dim_offsets[None, :]
    )
    out_mask = (query_offsets[:, None] < SEQ_LEN) & value_mask[None, :]
    tl.store(out_ptrs, output_values, mask=out_mask)


@overload
def flash_screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    return_score: Literal[False] = False,
    eps: float = 1e-6,
) -> torch.Tensor: ...


@overload
def flash_screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    return_score: Literal[True],
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]: ...


@overload
def flash_screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    return_score: bool,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]: ...


def flash_screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
    return_score: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Screening forward pass backed by a fused Triton kernel when possible.

    The CUDA path is intended for inference. If gradients are enabled for any
    differentiable input, or if the tensors are not supported by the kernel, the
    eager reference implementation is used to preserve autograd behavior.
    """

    _validate_inputs(
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    if _should_use_eager(
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
    ):
        return eager_screening(
            query=query,
            key=key,
            value=value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_score=return_score,
        )

    batch_size, num_heads, seq_len, key_dim = query.shape
    value_dim = value.size(-1)

    if position_ids is not None and position_ids.ndim == 3:
        position_ids = position_ids.squeeze(-1)

    query_c = query.contiguous()
    key_c = key.contiguous()
    value_c = value.contiguous()
    acceptance_c = acceptance.contiguous()

    if is_causal:
        window_c = window.contiguous()
    else:
        window_c = torch.empty(
            (num_heads,),
            device=query.device,
            dtype=acceptance.dtype,
        )

    position_ids_c = (
        position_ids.contiguous()
        if position_ids is not None
        else torch.empty((1,), device=query.device, dtype=torch.int64)
    )
    attention_mask_c = (
        attention_mask.to(device=query.device).contiguous()
        if attention_mask is not None
        else torch.empty((1,), device=query.device, dtype=query.dtype)
    )

    output = torch.empty(
        (batch_size, num_heads, seq_len, value_dim),
        device=value.device,
        dtype=value.dtype,
    )
    score = (
        torch.zeros(
            (batch_size, num_heads, seq_len, seq_len),
            device=query.device,
            dtype=query.dtype,
        )
        if return_score
        else output
    )

    block_m = 16
    block_n = 32
    block_dk = max(16, triton.next_power_of_2(key_dim))
    block_dv = max(16, triton.next_power_of_2(value_dim))
    num_warps = 4 if max(block_dk, block_dv) <= 128 else 8

    grid = (triton.cdiv(seq_len, block_m), batch_size * num_heads)
    kernel = cast(Any, _flash_screening_forward_kernel)
    kernel[grid](
        query_c,
        key_c,
        value_c,
        acceptance_c,
        window_c,
        position_ids_c,
        attention_mask_c,
        output,
        score,
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        HAS_POSITION_IDS=position_ids is not None,
        HAS_ATTENTION_MASK=attention_mask is not None,
        IS_CAUSAL=is_causal,
        RETURN_SCORE=return_score,
        EPS=eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DK=block_dk,
        BLOCK_DV=block_dv,
        num_warps=num_warps,
        num_stages=3,
    )

    if return_score:
        return output, score

    return output


def _validate_inputs(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
) -> None:
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("query, key, and value must be shaped [batch, heads, seq, dim]")

    if query.shape[:-1] != key.shape[:-1]:
        raise ValueError("query and key must share [batch, heads, seq] dimensions")

    if query.shape[:3] != value.shape[:3]:
        raise ValueError("value must share [batch, heads, seq] with query")

    if query.size(-1) != key.size(-1):
        raise ValueError("query and key must have the same feature dimension")

    if query.dtype != key.dtype or query.dtype != value.dtype:
        raise ValueError("query, key, and value must have the same dtype")

    if query.device != key.device or query.device != value.device:
        raise ValueError("query, key, and value must be on the same device")

    num_heads = query.size(1)
    if acceptance.shape != (num_heads,):
        raise ValueError("acceptance must be shaped [num_heads]")

    if window.shape != (num_heads,):
        raise ValueError("window must be shaped [num_heads]")

    if acceptance.device != query.device:
        raise ValueError("acceptance must be on the same device as query")

    if query.is_cuda and window.device != query.device:
        raise ValueError("window must be on the same device as query for CUDA execution")

    if not torch.is_floating_point(query):
        raise TypeError("query, key, and value must be floating-point tensors")

    if not torch.is_floating_point(acceptance) or not torch.is_floating_point(window):
        raise TypeError("acceptance and window must be floating-point tensors")

    batch_size, _, seq_len, _ = query.shape
    if position_ids is not None:
        valid_position_shape = position_ids.shape == (batch_size, seq_len) or (
            position_ids.shape == (batch_size, seq_len, 1)
        )
        if not valid_position_shape:
            raise ValueError(
                "position_ids must be shaped [batch, seq] or [batch, seq, 1]"
            )
        if position_ids.device != query.device:
            raise ValueError("position_ids must be on the same device as query")

    if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
        raise ValueError("attention_mask must be shaped [batch, seq]")


def _should_use_eager(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
) -> bool:
    if not query.is_cuda:
        return True

    if query.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return True

    if query.size(-1) > _MAX_KERNEL_BLOCK_D or value.size(-1) > _MAX_KERNEL_BLOCK_D:
        return True

    differentiable_inputs = (query, key, value, acceptance, window)
    if torch.is_grad_enabled() and any(t.requires_grad for t in differentiable_inputs):
        return True

    return False
