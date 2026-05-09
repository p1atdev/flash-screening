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
    hidden,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    HAS_ATTENTION_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    RETURN_SCORE: tl.constexpr,
    RETURN_HIDDEN: tl.constexpr,
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
    if RETURN_HIDDEN:
        hidden_ptrs = (
            hidden
            + (bh_offset + query_offsets[:, None]) * VALUE_DIM
            + value_dim_offsets[None, :]
        )
        hidden_mask = (query_offsets[:, None] < SEQ_LEN) & value_mask[None, :]
        tl.store(hidden_ptrs, acc, mask=hidden_mask)

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


@triton.jit
def _flash_screening_backward_kernel(
    query,
    key,
    value,
    acceptance,
    window,
    position_ids,
    attention_mask,
    grad_output,
    hidden,
    d_query,
    d_key,
    d_value,
    d_acceptance,
    d_window,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    HAS_POSITION_IDS: tl.constexpr,
    HAS_ATTENTION_MASK: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    NEED_DQUERY: tl.constexpr,
    NEED_DKEY: tl.constexpr,
    NEED_DVALUE: tl.constexpr,
    NEED_DACCEPTANCE: tl.constexpr,
    NEED_DWINDOW: tl.constexpr,
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

    value_mask = value_dim_offsets < VALUE_DIM

    grad_output_ptrs = (
        grad_output
        + (bh_offset + query_offsets[:, None]) * VALUE_DIM
        + value_dim_offsets[None, :]
    )
    grad_output_mask = (query_offsets[:, None] < SEQ_LEN) & value_mask[None, :]
    grad_output_values = tl.load(
        grad_output_ptrs,
        mask=grad_output_mask,
        other=0.0,
    ).to(tl.float32)

    hidden_ptrs = (
        hidden
        + (bh_offset + query_offsets[:, None]) * VALUE_DIM
        + value_dim_offsets[None, :]
    )
    hidden_values = tl.load(hidden_ptrs, mask=grad_output_mask, other=0.0).to(
        tl.float32
    )

    hidden_squared = tl.where(value_mask[None, :], hidden_values * hidden_values, 0.0)
    hidden_norm = tl.sqrt(tl.sum(hidden_squared, axis=1))
    exp_neg_2_norm = tl.exp(-2.0 * hidden_norm)
    tanh_norm = (1.0 - exp_neg_2_norm) / (1.0 + exp_neg_2_norm)
    sech2_norm = 4.0 * exp_neg_2_norm / (
        (1.0 + exp_neg_2_norm) * (1.0 + exp_neg_2_norm)
    )
    scale = tanh_norm / tl.maximum(hidden_norm, EPS)
    dot_grad_hidden = tl.sum(grad_output_values * hidden_values, axis=1)
    norm_cubed = hidden_norm * hidden_norm * hidden_norm
    radial_scale = dot_grad_hidden * (
        hidden_norm * sech2_norm - tanh_norm
    ) / tl.maximum(norm_cubed, EPS)
    d_hidden = tl.where(
        hidden_norm[:, None] > EPS,
        scale[:, None] * grad_output_values + radial_scale[:, None] * hidden_values,
        grad_output_values,
    )

    q_ptrs = (
        query
        + (bh_offset + query_offsets[:, None]) * KEY_DIM
        + key_dim_offsets[None, :]
    )
    q_mask = (query_offsets[:, None] < SEQ_LEN) & (
        key_dim_offsets[None, :] < KEY_DIM
    )
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    d_query_acc = tl.zeros((BLOCK_M, BLOCK_DK), dtype=tl.float32)

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
        trim_base = 1.0 - (1.0 - similarity) * inv_acceptance_h
        trim_positive = tl.maximum(trim_base, 0.0)
        trim_active = trim_base > 0.0
        trim = trim_positive * trim_positive

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
            softmask_angle = 3.141592653589793 * position_diff / window_h
            softmask = 0.5 * (tl.cos(softmask_angle) + 1.0)
            softmask = tl.where(softmask_valid, softmask, 0.0)
            d_softmask_d_window = (
                0.5
                * tl.sin(softmask_angle)
                * 3.141592653589793
                * position_diff
                / (window_h * window_h)
            )
            d_softmask_d_window = tl.where(
                softmask_valid,
                d_softmask_d_window,
                0.0,
            )
        else:
            softmask = tl.full((BLOCK_M, BLOCK_N), 1.0, dtype=tl.float32)
            d_softmask_d_window = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

        relevance = trim * softmask
        relevance = tl.where(valid_relevance, relevance, 0.0)

        key_mask = tl.full((BLOCK_N,), 1.0, dtype=tl.float32)
        if HAS_ATTENTION_MASK:
            key_mask = tl.load(
                attention_mask + batch_idx * SEQ_LEN + key_indices,
                mask=key_indices < SEQ_LEN,
                other=0.0,
            ).to(tl.float32)
            relevance = relevance * key_mask[None, :]

        v_ptrs = (
            value
            + (bh_offset + key_indices[:, None]) * VALUE_DIM
            + value_dim_offsets[None, :]
        )
        v_mask = (key_indices[:, None] < SEQ_LEN) & (
            value_dim_offsets[None, :] < VALUE_DIM
        )
        v = tl.load(v_ptrs, mask=v_mask, other=0.0).to(tl.float32)

        d_relevance = tl.dot(d_hidden, tl.trans(v), input_precision="tf32")
        d_relevance = tl.where(valid_relevance, d_relevance, 0.0)

        if NEED_DVALUE:
            d_value_update = tl.dot(tl.trans(relevance), d_hidden, input_precision="tf32")
            d_value_ptrs = (
                d_value
                + (bh_offset + key_indices[:, None]) * VALUE_DIM
                + value_dim_offsets[None, :]
            )
            tl.atomic_add(d_value_ptrs, d_value_update, mask=v_mask, sem="relaxed")

        d_trim = d_relevance * softmask * key_mask[None, :]
        d_similarity = tl.where(
            trim_active,
            d_trim * 2.0 * trim_positive * inv_acceptance_h,
            0.0,
        )

        if NEED_DQUERY:
            d_query_acc += tl.dot(d_similarity, tl.trans(k), input_precision="tf32")

        if NEED_DKEY:
            d_key_update = tl.dot(tl.trans(d_similarity), q, input_precision="tf32")
            d_key_ptrs = (
                d_key
                + (bh_offset + key_indices[:, None]) * KEY_DIM
                + key_dim_offsets[None, :]
            )
            d_key_mask = (key_indices[:, None] < SEQ_LEN) & (
                key_dim_offsets[None, :] < KEY_DIM
            )
            tl.atomic_add(d_key_ptrs, d_key_update, mask=d_key_mask, sem="relaxed")

        if NEED_DACCEPTANCE:
            d_acceptance_values = tl.where(
                trim_active,
                d_trim
                * 2.0
                * trim_positive
                * (1.0 - similarity)
                * inv_acceptance_h
                * inv_acceptance_h,
                0.0,
            )
            d_acceptance_values = tl.where(
                valid_relevance,
                d_acceptance_values,
                0.0,
            )
            d_acceptance_sum = tl.sum(tl.sum(d_acceptance_values, axis=0), axis=0)
            tl.atomic_add(
                d_acceptance + head_idx,
                d_acceptance_sum,
                sem="relaxed",
            )

        if NEED_DWINDOW:
            d_window_values = (
                d_relevance * trim * key_mask[None, :] * d_softmask_d_window
            )
            d_window_values = tl.where(valid_relevance, d_window_values, 0.0)
            d_window_sum = tl.sum(tl.sum(d_window_values, axis=0), axis=0)
            tl.atomic_add(d_window + head_idx, d_window_sum, sem="relaxed")

        start_n += BLOCK_N

    if NEED_DQUERY:
        d_query_ptrs = (
            d_query
            + (bh_offset + query_offsets[:, None]) * KEY_DIM
            + key_dim_offsets[None, :]
        )
        tl.store(d_query_ptrs, d_query_acc, mask=q_mask)


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
    """Screening operation backed by fused Triton kernels when possible."""

    _validate_inputs(
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        position_ids=position_ids,
        attention_mask=attention_mask,
        is_causal=is_causal,
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

    unsupported_grad = torch.is_grad_enabled() and (
        (position_ids is not None and position_ids.requires_grad)
        or (attention_mask is not None and attention_mask.requires_grad)
    )
    if unsupported_grad:
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

    differentiable_inputs = (query, key, value, acceptance)
    if is_causal:
        differentiable_inputs = (*differentiable_inputs, window)

    needs_grad = torch.is_grad_enabled() and _requires_grad(*differentiable_inputs)
    if needs_grad:
        if return_score:
            return eager_screening(
                query=query,
                key=key,
                value=value,
                acceptance=acceptance,
                window=window,
                position_ids=position_ids,
                attention_mask=attention_mask,
                is_causal=is_causal,
                return_score=True,
            )

        return _FlashScreeningFunction.apply(
            query,
            key,
            value,
            acceptance,
            window,
            position_ids,
            attention_mask,
            is_causal,
            eps,
        )

    output, score, _, _ = _run_flash_screening_forward(
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        position_ids=position_ids,
        attention_mask=attention_mask,
        is_causal=is_causal,
        return_score=return_score,
        return_hidden=False,
        eps=eps,
    )

    if return_score:
        if score is None:
            raise RuntimeError("flash_screening did not produce a score tensor")
        return output, score

    return output


class _FlashScreeningFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        acceptance: torch.Tensor,
        window: torch.Tensor,
        position_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        is_causal: bool,
        eps: float,
    ) -> torch.Tensor:
        output, _, hidden, saved_tensors = _run_flash_screening_forward(
            query=query,
            key=key,
            value=value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_causal=is_causal,
            return_score=False,
            return_hidden=True,
            eps=eps,
        )
        if hidden is None:
            raise RuntimeError("flash_screening did not produce a hidden tensor")

        ctx.save_for_backward(*saved_tensors, hidden)
        ctx.has_position_ids = position_ids is not None
        ctx.has_attention_mask = attention_mask is not None
        ctx.is_causal = is_causal
        ctx.eps = eps
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Any:
        grad_output = cast(torch.Tensor, grad_outputs[0])
        (
            query,
            key,
            value,
            acceptance,
            window,
            position_ids,
            attention_mask,
            hidden,
        ) = ctx.saved_tensors

        needs = ctx.needs_input_grad
        need_dquery = needs[0]
        need_dkey = needs[1]
        need_dvalue = needs[2]
        need_dacceptance = needs[3]
        need_dwindow = needs[4] and ctx.is_causal

        d_query = torch.empty_like(query)
        d_key = torch.zeros_like(key)
        d_value = torch.zeros_like(value)
        d_acceptance = torch.zeros_like(acceptance)
        d_window = torch.zeros_like(window)

        _run_flash_screening_backward(
            query=query,
            key=key,
            value=value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
            grad_output=grad_output.contiguous(),
            hidden=hidden,
            d_query=d_query,
            d_key=d_key,
            d_value=d_value,
            d_acceptance=d_acceptance,
            d_window=d_window,
            has_position_ids=ctx.has_position_ids,
            has_attention_mask=ctx.has_attention_mask,
            is_causal=ctx.is_causal,
            need_dquery=need_dquery,
            need_dkey=need_dkey,
            need_dvalue=need_dvalue,
            need_dacceptance=need_dacceptance,
            need_dwindow=need_dwindow,
            eps=ctx.eps,
        )

        return (
            d_query if need_dquery else None,
            d_key if need_dkey else None,
            d_value if need_dvalue else None,
            d_acceptance if need_dacceptance else None,
            d_window if need_dwindow else None,
            None,
            None,
            None,
            None,
        )


def _run_flash_screening_forward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
    return_score: bool,
    return_hidden: bool,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor | None,
    torch.Tensor | None,
    tuple[torch.Tensor, ...],
]:
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
        else None
    )
    hidden = (
        torch.empty(
            (batch_size, num_heads, seq_len, value_dim),
            device=value.device,
            dtype=torch.float32,
        )
        if return_hidden
        else None
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
        score if score is not None else output,
        hidden if hidden is not None else output,
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        HAS_POSITION_IDS=position_ids is not None,
        HAS_ATTENTION_MASK=attention_mask is not None,
        IS_CAUSAL=is_causal,
        RETURN_SCORE=return_score,
        RETURN_HIDDEN=return_hidden,
        EPS=eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DK=block_dk,
        BLOCK_DV=block_dv,
        num_warps=num_warps,
        num_stages=3,
    )

    return (
        output,
        score,
        hidden,
        (
            query_c,
            key_c,
            value_c,
            acceptance_c,
            window_c,
            position_ids_c,
            attention_mask_c,
        ),
    )


def _run_flash_screening_backward(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grad_output: torch.Tensor,
    hidden: torch.Tensor,
    d_query: torch.Tensor,
    d_key: torch.Tensor,
    d_value: torch.Tensor,
    d_acceptance: torch.Tensor,
    d_window: torch.Tensor,
    has_position_ids: bool,
    has_attention_mask: bool,
    is_causal: bool,
    need_dquery: bool,
    need_dkey: bool,
    need_dvalue: bool,
    need_dacceptance: bool,
    need_dwindow: bool,
    eps: float,
) -> None:
    batch_size, num_heads, seq_len, key_dim = query.shape
    value_dim = value.size(-1)
    block_m = 16
    block_n = 32
    block_dk = max(16, triton.next_power_of_2(key_dim))
    block_dv = max(16, triton.next_power_of_2(value_dim))
    num_warps = 4 if max(block_dk, block_dv) <= 128 else 8

    grid = (triton.cdiv(seq_len, block_m), batch_size * num_heads)
    kernel = cast(Any, _flash_screening_backward_kernel)
    kernel[grid](
        query,
        key,
        value,
        acceptance,
        window,
        position_ids,
        attention_mask,
        grad_output,
        hidden,
        d_query,
        d_key,
        d_value,
        d_acceptance,
        d_window,
        NUM_HEADS=num_heads,
        SEQ_LEN=seq_len,
        KEY_DIM=key_dim,
        VALUE_DIM=value_dim,
        HAS_POSITION_IDS=has_position_ids,
        HAS_ATTENTION_MASK=has_attention_mask,
        IS_CAUSAL=is_causal,
        NEED_DQUERY=need_dquery,
        NEED_DKEY=need_dkey,
        NEED_DVALUE=need_dvalue,
        NEED_DACCEPTANCE=need_dacceptance,
        NEED_DWINDOW=need_dwindow,
        EPS=eps,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_DK=block_dk,
        BLOCK_DV=block_dv,
        num_warps=num_warps,
        num_stages=3,
    )


def _validate_inputs(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    position_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    is_causal: bool,
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
        if is_causal:
            valid_position_shape = position_ids.shape == (batch_size, seq_len) or (
                position_ids.shape == (batch_size, seq_len, 1)
            )
            if not valid_position_shape:
                raise ValueError(
                    "causal position_ids must be shaped [batch, seq] "
                    "or [batch, seq, 1]"
                )
        else:
            valid_position_shape = (
                position_ids.ndim >= 2
                and position_ids.shape[0] == batch_size
                and position_ids.shape[1] == seq_len
            )
            if not valid_position_shape:
                raise ValueError(
                    "non-causal position_ids must start with [batch, seq]"
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

    return False


def _requires_grad(*tensors: torch.Tensor) -> bool:
    return any(tensor.requires_grad for tensor in tensors)
