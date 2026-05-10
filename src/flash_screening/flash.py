from __future__ import annotations

from typing import Any, Literal, overload, cast

import torch
import triton
import triton.language as tl

from .eager import (
    apply_mipe as eager_apply_mipe,
    compute_freqs_cis as eager_compute_freqs_cis,
    mipe_rotation as eager_mipe_rotation,
    screening as eager_screening,
    unit_length_norm as eager_unit_length_norm,
)


_MAX_KERNEL_BLOCK_D = 256
_SUPPORTED_KERNEL_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


@triton.jit
def _unit_length_norm_forward_kernel(
    x,
    output,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    values = tl.load(x + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
    squared = tl.where(mask, values * values, 0.0)
    norm = tl.sqrt(tl.sum(squared, axis=0))
    denom = tl.maximum(norm, EPS)

    tl.store(output + row * D + offsets, values / denom, mask=mask)


@triton.jit
def _unit_length_norm_backward_kernel(
    x,
    grad_output,
    d_x,
    D: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_D)
    mask = offsets < D

    values = tl.load(x + row * D + offsets, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_output + row * D + offsets, mask=mask, other=0.0).to(tl.float32)

    squared = tl.where(mask, values * values, 0.0)
    norm = tl.sqrt(tl.sum(squared, axis=0))
    denom = tl.maximum(norm, EPS)
    dot = tl.sum(tl.where(mask, grad * values, 0.0), axis=0)

    d_values = tl.where(
        norm > EPS,
        grad / denom - values * dot / (denom * denom * denom),
        grad / EPS,
    )
    tl.store(d_x + row * D + offsets, d_values, mask=mask)


@triton.jit
def _mipe_rotation_forward_kernel(
    position_ids,
    window,
    output,
    TOTAL: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    WINDOW_THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    seq_idx = offsets % SEQ_LEN
    head_batch_idx = offsets // SEQ_LEN
    head_idx = head_batch_idx % NUM_HEADS
    batch_idx = head_batch_idx // NUM_HEADS

    pos = tl.load(
        position_ids + batch_idx * SEQ_LEN + seq_idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    window_h = tl.load(window + head_idx, mask=mask, other=1.0).to(tl.float32)

    active = window_h < WINDOW_THRESHOLD
    gamma = 0.5 * (tl.cos(3.141592653589793 * window_h / WINDOW_THRESHOLD) + 1.0)
    gamma = tl.where(active, gamma, 0.0)
    rotation = 3.141592653589793 * pos * gamma / window_h

    tl.store(output + offsets, rotation, mask=mask)


@triton.jit
def _mipe_rotation_backward_kernel(
    position_ids,
    window,
    grad_output,
    d_position_ids,
    d_window,
    TOTAL: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NEED_DPOSITION_IDS: tl.constexpr,
    NEED_DWINDOW: tl.constexpr,
    WINDOW_THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    seq_idx = offsets % SEQ_LEN
    head_batch_idx = offsets // SEQ_LEN
    head_idx = head_batch_idx % NUM_HEADS
    batch_idx = head_batch_idx // NUM_HEADS

    pos = tl.load(
        position_ids + batch_idx * SEQ_LEN + seq_idx,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    window_h = tl.load(window + head_idx, mask=mask, other=1.0).to(tl.float32)
    grad = tl.load(grad_output + offsets, mask=mask, other=0.0).to(tl.float32)

    active = window_h < WINDOW_THRESHOLD
    gamma = 0.5 * (tl.cos(3.141592653589793 * window_h / WINDOW_THRESHOLD) + 1.0)
    gamma = tl.where(active, gamma, 0.0)
    dgamma = (
        -0.5
        * tl.sin(3.141592653589793 * window_h / WINDOW_THRESHOLD)
        * (3.141592653589793 / WINDOW_THRESHOLD)
    )
    dgamma = tl.where(active, dgamma, 0.0)

    coeff = 3.141592653589793 * gamma / window_h
    dcoeff = 3.141592653589793 * (dgamma * window_h - gamma) / (window_h * window_h)

    if NEED_DPOSITION_IDS:
        tl.atomic_add(
            d_position_ids + batch_idx * SEQ_LEN + seq_idx,
            grad * coeff,
            mask=mask,
            sem="relaxed",
        )

    if NEED_DWINDOW:
        tl.atomic_add(
            d_window + head_idx,
            grad * pos * dcoeff,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def _compute_freqs_cis_forward_kernel(
    position_ids,
    window,
    output,
    TOTAL: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_AXES: tl.constexpr,
    POSITION_IDS_NDIM: tl.constexpr,
    WINDOW_THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    axis_idx = offsets % NUM_AXES
    seq_axis_idx = offsets // NUM_AXES
    seq_idx = seq_axis_idx % SEQ_LEN
    head_seq_axis_idx = seq_axis_idx // SEQ_LEN
    head_idx = head_seq_axis_idx % NUM_HEADS
    batch_idx = head_seq_axis_idx // NUM_HEADS

    if POSITION_IDS_NDIM == 2:
        pos_offsets = batch_idx * SEQ_LEN + seq_idx
    else:
        pos_offsets = (batch_idx * SEQ_LEN + seq_idx) * NUM_AXES + axis_idx

    pos = tl.load(position_ids + pos_offsets, mask=mask, other=0.0).to(tl.float32)
    window_h = tl.load(window + head_idx, mask=mask, other=1.0).to(tl.float32)

    active = window_h < WINDOW_THRESHOLD
    gamma = 0.5 * (tl.cos(3.141592653589793 * window_h / WINDOW_THRESHOLD) + 1.0)
    gamma = tl.where(active, gamma, 0.0)
    rotation = 3.141592653589793 * pos * gamma / window_h

    out_base = ((batch_idx * NUM_HEADS + head_idx) * SEQ_LEN + seq_idx) * (
        2 * NUM_AXES
    ) + 2 * axis_idx
    tl.store(output + out_base, tl.cos(rotation), mask=mask)
    tl.store(output + out_base + 1, tl.sin(rotation), mask=mask)


@triton.jit
def _compute_freqs_cis_backward_kernel(
    position_ids,
    window,
    grad_output,
    d_position_ids,
    d_window,
    TOTAL: tl.constexpr,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    NUM_AXES: tl.constexpr,
    POSITION_IDS_NDIM: tl.constexpr,
    NEED_DPOSITION_IDS: tl.constexpr,
    NEED_DWINDOW: tl.constexpr,
    WINDOW_THRESHOLD: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    axis_idx = offsets % NUM_AXES
    seq_axis_idx = offsets // NUM_AXES
    seq_idx = seq_axis_idx % SEQ_LEN
    head_seq_axis_idx = seq_axis_idx // SEQ_LEN
    head_idx = head_seq_axis_idx % NUM_HEADS
    batch_idx = head_seq_axis_idx // NUM_HEADS

    if POSITION_IDS_NDIM == 2:
        pos_offsets = batch_idx * SEQ_LEN + seq_idx
    else:
        pos_offsets = (batch_idx * SEQ_LEN + seq_idx) * NUM_AXES + axis_idx

    pos = tl.load(position_ids + pos_offsets, mask=mask, other=0.0).to(tl.float32)
    window_h = tl.load(window + head_idx, mask=mask, other=1.0).to(tl.float32)

    active = window_h < WINDOW_THRESHOLD
    gamma = 0.5 * (tl.cos(3.141592653589793 * window_h / WINDOW_THRESHOLD) + 1.0)
    gamma = tl.where(active, gamma, 0.0)
    dgamma = (
        -0.5
        * tl.sin(3.141592653589793 * window_h / WINDOW_THRESHOLD)
        * (3.141592653589793 / WINDOW_THRESHOLD)
    )
    dgamma = tl.where(active, dgamma, 0.0)

    coeff = 3.141592653589793 * gamma / window_h
    dcoeff = 3.141592653589793 * (dgamma * window_h - gamma) / (window_h * window_h)
    rotation = pos * coeff

    out_base = ((batch_idx * NUM_HEADS + head_idx) * SEQ_LEN + seq_idx) * (
        2 * NUM_AXES
    ) + 2 * axis_idx
    grad_cos = tl.load(grad_output + out_base, mask=mask, other=0.0).to(tl.float32)
    grad_sin = tl.load(grad_output + out_base + 1, mask=mask, other=0.0).to(tl.float32)
    d_rotation = -grad_cos * tl.sin(rotation) + grad_sin * tl.cos(rotation)

    if NEED_DPOSITION_IDS:
        tl.atomic_add(
            d_position_ids + pos_offsets,
            d_rotation * coeff,
            mask=mask,
            sem="relaxed",
        )

    if NEED_DWINDOW:
        tl.atomic_add(
            d_window + head_idx,
            d_rotation * pos * dcoeff,
            mask=mask,
            sem="relaxed",
        )


@triton.jit
def _apply_mipe_forward_kernel(
    sequence,
    freqs_cis,
    output,
    TOTAL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ENCODED_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL

    dim_idx = offsets % HEAD_DIM
    token_idx = offsets // HEAD_DIM
    in_encoded = dim_idx < ENCODED_DIM
    pair_dim = (dim_idx // 2) * 2

    seq_base = token_idx * HEAD_DIM
    freq_base = token_idx * ENCODED_DIM
    x_even = tl.load(sequence + seq_base + pair_dim, mask=mask & in_encoded, other=0.0)
    x_odd = tl.load(
        sequence + seq_base + pair_dim + 1,
        mask=mask & in_encoded,
        other=0.0,
    )
    cos = tl.load(freqs_cis + freq_base + pair_dim, mask=mask & in_encoded, other=0.0)
    sin = tl.load(
        freqs_cis + freq_base + pair_dim + 1,
        mask=mask & in_encoded,
        other=0.0,
    )

    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos
    rotated = tl.where(dim_idx % 2 == 0, rotated_even, rotated_odd)

    tail = tl.load(sequence + offsets, mask=mask & (dim_idx >= ENCODED_DIM), other=0.0)
    values = tl.where(in_encoded, rotated, tail)
    tl.store(output + offsets, values, mask=mask)


@triton.jit
def _apply_mipe_backward_pairs_kernel(
    sequence,
    freqs_cis,
    grad_output,
    d_sequence,
    d_freqs_cis,
    TOTAL_PAIRS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ENCODED_DIM: tl.constexpr,
    NUM_PAIRS: tl.constexpr,
    NEED_DSEQUENCE: tl.constexpr,
    NEED_DFREQS_CIS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL_PAIRS

    pair_idx = offsets % NUM_PAIRS
    token_idx = offsets // NUM_PAIRS
    seq_base = token_idx * HEAD_DIM + 2 * pair_idx
    freq_base = token_idx * ENCODED_DIM + 2 * pair_idx

    x_even = tl.load(sequence + seq_base, mask=mask, other=0.0).to(tl.float32)
    x_odd = tl.load(sequence + seq_base + 1, mask=mask, other=0.0).to(tl.float32)
    cos = tl.load(freqs_cis + freq_base, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(freqs_cis + freq_base + 1, mask=mask, other=0.0).to(tl.float32)
    grad_even = tl.load(grad_output + seq_base, mask=mask, other=0.0).to(tl.float32)
    grad_odd = tl.load(grad_output + seq_base + 1, mask=mask, other=0.0).to(tl.float32)

    if NEED_DSEQUENCE:
        d_even = grad_even * cos + grad_odd * sin
        d_odd = -grad_even * sin + grad_odd * cos
        tl.store(d_sequence + seq_base, d_even, mask=mask)
        tl.store(d_sequence + seq_base + 1, d_odd, mask=mask)

    if NEED_DFREQS_CIS:
        d_cos = grad_even * x_even + grad_odd * x_odd
        d_sin = -grad_even * x_odd + grad_odd * x_even
        tl.store(d_freqs_cis + freq_base, d_cos, mask=mask)
        tl.store(d_freqs_cis + freq_base + 1, d_sin, mask=mask)


@triton.jit
def _apply_mipe_backward_tail_kernel(
    grad_output,
    d_sequence,
    TOTAL_TAIL: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ENCODED_DIM: tl.constexpr,
    TAIL_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < TOTAL_TAIL

    tail_idx = offsets % TAIL_DIM
    token_idx = offsets // TAIL_DIM
    dim_idx = ENCODED_DIM + tail_idx
    seq_offset = token_idx * HEAD_DIM + dim_idx

    grad = tl.load(grad_output + seq_offset, mask=mask, other=0.0)
    tl.store(d_sequence + seq_offset, grad, mask=mask)


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
    q_mask = (query_offsets[:, None] < SEQ_LEN) & (key_dim_offsets[None, :] < KEY_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_M, BLOCK_DV), dtype=tl.float32)
    acceptance_h = tl.load(acceptance + head_idx).to(tl.float32)
    inv_acceptance_h = 1.0 / acceptance_h

    if IS_CAUSAL:
        window_h = tl.load(window + head_idx).to(tl.float32)
    else:
        window_h = 1.0

    if IS_CAUSAL and not HAS_POSITION_IDS:
        window_span = (
            tl.minimum(
                window_h,
                tl.full((), SEQ_LEN + BLOCK_M, dtype=tl.float32),
            ).to(tl.int64)
            + 1
        )
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
        k_mask = (key_indices[None, :] < SEQ_LEN) & (key_dim_offsets[:, None] < KEY_DIM)
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
            softmask = 0.5 * (
                tl.cos(3.141592653589793 * position_diff / window_h) + 1.0
            )
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
    sech2_norm = (
        4.0 * exp_neg_2_norm / ((1.0 + exp_neg_2_norm) * (1.0 + exp_neg_2_norm))
    )
    scale = tanh_norm / tl.maximum(hidden_norm, EPS)
    dot_grad_hidden = tl.sum(grad_output_values * hidden_values, axis=1)
    norm_cubed = hidden_norm * hidden_norm * hidden_norm
    radial_scale = (
        dot_grad_hidden
        * (hidden_norm * sech2_norm - tanh_norm)
        / tl.maximum(norm_cubed, EPS)
    )
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
    q_mask = (query_offsets[:, None] < SEQ_LEN) & (key_dim_offsets[None, :] < KEY_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0).to(tl.float32)
    d_query_acc = tl.zeros((BLOCK_M, BLOCK_DK), dtype=tl.float32)

    acceptance_h = tl.load(acceptance + head_idx).to(tl.float32)
    inv_acceptance_h = 1.0 / acceptance_h

    if IS_CAUSAL:
        window_h = tl.load(window + head_idx).to(tl.float32)
    else:
        window_h = 1.0

    if IS_CAUSAL and not HAS_POSITION_IDS:
        window_span = (
            tl.minimum(
                window_h,
                tl.full((), SEQ_LEN + BLOCK_M, dtype=tl.float32),
            ).to(tl.int64)
            + 1
        )
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
        k_mask = (key_indices[None, :] < SEQ_LEN) & (key_dim_offsets[:, None] < KEY_DIM)
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
            d_value_update = tl.dot(
                tl.trans(relevance), d_hidden, input_precision="tf32"
            )
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


def unit_length_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Unit-length normalization backed by Triton kernels when possible."""

    if _should_use_eager_unit_length_norm(x):
        return eager_unit_length_norm(x, eps=eps)

    return _UnitLengthNormFunction.apply(x, eps)


def mipe_rotation(
    position_ids: torch.Tensor,
    window: torch.Tensor,
    window_threshold: float = 256.0,
) -> torch.Tensor:
    """MiPE rotation angles backed by Triton kernels when possible."""

    _validate_mipe_rotation_inputs(position_ids=position_ids, window=window)
    if _should_use_eager_mipe(position_ids=position_ids, window=window):
        return eager_mipe_rotation(
            position_ids=position_ids,
            window=window,
            window_threshold=window_threshold,
        )

    return _MipeRotationFunction.apply(position_ids, window, float(window_threshold))


def compute_freqs_cis(
    position_ids: torch.Tensor,
    window: torch.Tensor,
    window_threshold: float = 256.0,
) -> torch.Tensor:
    """MiPE cosine/sine frequencies backed by Triton kernels when possible."""

    _validate_compute_freqs_cis_inputs(position_ids=position_ids, window=window)
    if _should_use_eager_mipe(position_ids=position_ids, window=window):
        return eager_compute_freqs_cis(
            position_ids=position_ids,
            window=window,
            window_threshold=window_threshold,
        )

    return _ComputeFreqsCisFunction.apply(
        position_ids,
        window,
        float(window_threshold),
    )


def apply_mipe(sequence: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Apply MiPE rotations backed by Triton kernels when possible."""

    _validate_apply_mipe_inputs(sequence=sequence, freqs_cis=freqs_cis)
    if _should_use_eager_apply_mipe(sequence=sequence, freqs_cis=freqs_cis):
        return eager_apply_mipe(sequence=sequence, freqs_cis=freqs_cis)

    return _ApplyMipeFunction.apply(sequence, freqs_cis)


class _UnitLengthNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, eps: float) -> torch.Tensor:
        x_c = x.contiguous()
        output = torch.empty_like(x_c)
        rows = x_c.numel() // x_c.size(-1)
        block_d = max(16, triton.next_power_of_2(x_c.size(-1)))

        kernel = cast(Any, _unit_length_norm_forward_kernel)
        kernel[(rows,)](
            x_c,
            output,
            D=x_c.size(-1),
            EPS=eps,
            BLOCK_D=block_d,
            num_warps=1,
        )

        ctx.save_for_backward(x_c)
        ctx.eps = eps
        return output

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        grad_output = cast(torch.Tensor, grad_outputs[0])
        (x,) = ctx.saved_tensors
        grad_output_c = grad_output.contiguous()
        d_x = torch.empty_like(x)
        rows = x.numel() // x.size(-1)
        block_d = max(16, triton.next_power_of_2(x.size(-1)))

        kernel = cast(Any, _unit_length_norm_backward_kernel)
        kernel[(rows,)](
            x,
            grad_output_c,
            d_x,
            D=x.size(-1),
            EPS=ctx.eps,
            BLOCK_D=block_d,
            num_warps=1,
        )

        return d_x, None


class _MipeRotationFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        position_ids: torch.Tensor,
        window: torch.Tensor,
        window_threshold: float,
    ) -> torch.Tensor:
        batch_size, seq_len = position_ids.shape
        num_heads = window.size(0)
        position_ids_c = position_ids.contiguous()
        window_c = window.contiguous()
        output = torch.empty(
            (batch_size, num_heads, seq_len),
            device=window.device,
            dtype=torch.result_type(position_ids, window),
        )
        total = output.numel()

        kernel = cast(Any, _mipe_rotation_forward_kernel)
        kernel[(triton.cdiv(total, 256),)](
            position_ids_c,
            window_c,
            output,
            TOTAL=total,
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            WINDOW_THRESHOLD=window_threshold,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        ctx.save_for_backward(position_ids_c, window_c)
        ctx.window_threshold = window_threshold
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Any:
        grad_output = cast(torch.Tensor, grad_outputs[0])
        position_ids, window = ctx.saved_tensors
        batch_size, seq_len = position_ids.shape
        num_heads = window.size(0)
        total = batch_size * num_heads * seq_len
        needs = ctx.needs_input_grad
        need_dposition_ids = needs[0]
        need_dwindow = needs[1]

        d_position_ids = (
            torch.zeros_like(position_ids)
            if need_dposition_ids
            else torch.empty((1,), device=window.device, dtype=window.dtype)
        )
        d_window = (
            torch.zeros_like(window)
            if need_dwindow
            else torch.empty((1,), device=window.device, dtype=window.dtype)
        )

        kernel = cast(Any, _mipe_rotation_backward_kernel)
        kernel[(triton.cdiv(total, 256),)](
            position_ids,
            window,
            grad_output.contiguous(),
            d_position_ids,
            d_window,
            TOTAL=total,
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            NEED_DPOSITION_IDS=need_dposition_ids,
            NEED_DWINDOW=need_dwindow,
            WINDOW_THRESHOLD=ctx.window_threshold,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        return (
            d_position_ids if need_dposition_ids else None,
            d_window if need_dwindow else None,
            None,
        )


class _ComputeFreqsCisFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        position_ids: torch.Tensor,
        window: torch.Tensor,
        window_threshold: float,
    ) -> torch.Tensor:
        batch_size, seq_len = position_ids.shape[:2]
        num_axes = 1 if position_ids.ndim == 2 else position_ids.size(-1)
        num_heads = window.size(0)
        position_ids_c = position_ids.contiguous()
        window_c = window.contiguous()
        output = torch.empty(
            (batch_size, num_heads, seq_len, 2 * num_axes),
            device=window.device,
            dtype=torch.result_type(position_ids, window),
        )
        total = batch_size * num_heads * seq_len * num_axes

        kernel = cast(Any, _compute_freqs_cis_forward_kernel)
        kernel[(triton.cdiv(total, 256),)](
            position_ids_c,
            window_c,
            output,
            TOTAL=total,
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            NUM_AXES=num_axes,
            POSITION_IDS_NDIM=position_ids.ndim,
            WINDOW_THRESHOLD=window_threshold,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        ctx.save_for_backward(position_ids_c, window_c)
        ctx.num_axes = num_axes
        ctx.position_ids_ndim = position_ids.ndim
        ctx.window_threshold = window_threshold
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Any:
        grad_output = cast(torch.Tensor, grad_outputs[0])
        position_ids, window = ctx.saved_tensors
        batch_size, seq_len = position_ids.shape[:2]
        num_heads = window.size(0)
        total = batch_size * num_heads * seq_len * ctx.num_axes
        needs = ctx.needs_input_grad
        need_dposition_ids = needs[0]
        need_dwindow = needs[1]

        d_position_ids = (
            torch.zeros_like(position_ids)
            if need_dposition_ids
            else torch.empty((1,), device=window.device, dtype=window.dtype)
        )
        d_window = (
            torch.zeros_like(window)
            if need_dwindow
            else torch.empty((1,), device=window.device, dtype=window.dtype)
        )

        kernel = cast(Any, _compute_freqs_cis_backward_kernel)
        kernel[(triton.cdiv(total, 256),)](
            position_ids,
            window,
            grad_output.contiguous(),
            d_position_ids,
            d_window,
            TOTAL=total,
            NUM_HEADS=num_heads,
            SEQ_LEN=seq_len,
            NUM_AXES=ctx.num_axes,
            POSITION_IDS_NDIM=ctx.position_ids_ndim,
            NEED_DPOSITION_IDS=need_dposition_ids,
            NEED_DWINDOW=need_dwindow,
            WINDOW_THRESHOLD=ctx.window_threshold,
            BLOCK_SIZE=256,
            num_warps=4,
        )

        return (
            d_position_ids if need_dposition_ids else None,
            d_window if need_dwindow else None,
            None,
        )


class _ApplyMipeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        sequence: torch.Tensor,
        freqs_cis: torch.Tensor,
    ) -> torch.Tensor:
        sequence_c = sequence.contiguous()
        freqs_cis_c = freqs_cis.contiguous()
        output = torch.empty(
            sequence_c.shape,
            device=sequence.device,
            dtype=torch.result_type(sequence, freqs_cis),
        )
        total = output.numel()

        kernel = cast(Any, _apply_mipe_forward_kernel)
        kernel[(triton.cdiv(total, 256),)](
            sequence_c,
            freqs_cis_c,
            output,
            TOTAL=total,
            HEAD_DIM=sequence_c.size(-1),
            ENCODED_DIM=freqs_cis_c.size(-1),
            BLOCK_SIZE=256,
            num_warps=4,
        )

        ctx.save_for_backward(sequence_c, freqs_cis_c)
        return output

    @staticmethod
    def backward(
        ctx: Any,
        *grad_outputs: Any,
    ) -> Any:
        grad_output = cast(torch.Tensor, grad_outputs[0])
        sequence, freqs_cis = ctx.saved_tensors
        needs = ctx.needs_input_grad
        need_dsequence = needs[0]
        need_dfreqs_cis = needs[1]
        d_sequence = (
            torch.empty_like(sequence)
            if need_dsequence
            else torch.empty((1,), device=sequence.device, dtype=sequence.dtype)
        )
        d_freqs_cis = (
            torch.empty_like(freqs_cis)
            if need_dfreqs_cis
            else torch.empty((1,), device=freqs_cis.device, dtype=freqs_cis.dtype)
        )

        _run_apply_mipe_backward(
            sequence=sequence,
            freqs_cis=freqs_cis,
            grad_output=grad_output.contiguous(),
            d_sequence=d_sequence,
            d_freqs_cis=d_freqs_cis,
            need_dsequence=need_dsequence,
            need_dfreqs_cis=need_dfreqs_cis,
        )

        return (
            d_sequence if need_dsequence else None,
            d_freqs_cis if need_dfreqs_cis else None,
        )


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


def _run_apply_mipe_backward(
    *,
    sequence: torch.Tensor,
    freqs_cis: torch.Tensor,
    grad_output: torch.Tensor,
    d_sequence: torch.Tensor,
    d_freqs_cis: torch.Tensor,
    need_dsequence: bool,
    need_dfreqs_cis: bool,
) -> None:
    head_dim = sequence.size(-1)
    encoded_dim = freqs_cis.size(-1)
    token_count = sequence.numel() // head_dim
    num_pairs = encoded_dim // 2

    if num_pairs > 0:
        total_pairs = token_count * num_pairs
        pairs_kernel = cast(Any, _apply_mipe_backward_pairs_kernel)
        pairs_kernel[(triton.cdiv(total_pairs, 256),)](
            sequence,
            freqs_cis,
            grad_output,
            d_sequence,
            d_freqs_cis,
            TOTAL_PAIRS=total_pairs,
            HEAD_DIM=head_dim,
            ENCODED_DIM=encoded_dim,
            NUM_PAIRS=num_pairs,
            NEED_DSEQUENCE=need_dsequence,
            NEED_DFREQS_CIS=need_dfreqs_cis,
            BLOCK_SIZE=256,
            num_warps=4,
        )

    tail_dim = head_dim - encoded_dim
    if need_dsequence and tail_dim > 0:
        total_tail = token_count * tail_dim
        tail_kernel = cast(Any, _apply_mipe_backward_tail_kernel)
        tail_kernel[(triton.cdiv(total_tail, 256),)](
            grad_output,
            d_sequence,
            TOTAL_TAIL=total_tail,
            HEAD_DIM=head_dim,
            ENCODED_DIM=encoded_dim,
            TAIL_DIM=tail_dim,
            BLOCK_SIZE=256,
            num_warps=4,
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
        raise ValueError(
            "query, key, and value must be shaped [batch, heads, seq, dim]"
        )

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
        raise ValueError(
            "window must be on the same device as query for CUDA execution"
        )

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
                    "causal position_ids must be shaped [batch, seq] or [batch, seq, 1]"
                )
        else:
            valid_position_shape = (
                position_ids.ndim >= 2
                and position_ids.shape[0] == batch_size
                and position_ids.shape[1] == seq_len
            )
            if not valid_position_shape:
                raise ValueError("non-causal position_ids must start with [batch, seq]")
        if position_ids.device != query.device:
            raise ValueError("position_ids must be on the same device as query")

    if attention_mask is not None and attention_mask.shape != (batch_size, seq_len):
        raise ValueError("attention_mask must be shaped [batch, seq]")


def _validate_mipe_rotation_inputs(
    *,
    position_ids: torch.Tensor,
    window: torch.Tensor,
) -> None:
    if position_ids.ndim != 2:
        raise ValueError("position_ids must be shaped [batch, seq]")

    if window.ndim != 1:
        raise ValueError("window must be shaped [num_heads]")

    if position_ids.device != window.device:
        raise ValueError("position_ids and window must be on the same device")

    if not torch.is_floating_point(window):
        raise TypeError("window must be a floating-point tensor")


def _validate_compute_freqs_cis_inputs(
    *,
    position_ids: torch.Tensor,
    window: torch.Tensor,
) -> None:
    if position_ids.ndim not in (2, 3):
        raise ValueError(
            "position_ids must be shaped [batch, seq] or [batch, seq, axes]"
        )

    if position_ids.ndim == 3 and position_ids.size(-1) == 0:
        raise ValueError("position_ids must contain at least one axis")

    if window.ndim != 1:
        raise ValueError("window must be shaped [num_heads]")

    if position_ids.device != window.device:
        raise ValueError("position_ids and window must be on the same device")

    if not torch.is_floating_point(window):
        raise TypeError("window must be a floating-point tensor")


def _validate_apply_mipe_inputs(
    *,
    sequence: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> None:
    if sequence.ndim != 4 or freqs_cis.ndim != 4:
        raise ValueError(
            "sequence and freqs_cis must be shaped [batch, heads, seq, dim]"
        )

    if sequence.shape[:3] != freqs_cis.shape[:3]:
        raise ValueError("sequence and freqs_cis must share [batch, heads, seq]")

    encoded_dim = freqs_cis.size(-1)
    if encoded_dim % 2 != 0:
        raise ValueError("encoded_dim must be even")

    if sequence.size(-1) < encoded_dim:
        raise ValueError("head_dim must be >= encoded_dim")

    if sequence.device != freqs_cis.device:
        raise ValueError("sequence and freqs_cis must be on the same device")

    if not torch.is_floating_point(sequence) or not torch.is_floating_point(freqs_cis):
        raise TypeError("sequence and freqs_cis must be floating-point tensors")


def _should_use_eager_unit_length_norm(x: torch.Tensor) -> bool:
    if x.ndim == 0 or x.numel() == 0:
        return True

    if not x.is_cuda:
        return True

    if not torch.is_floating_point(x) or x.dtype not in _SUPPORTED_KERNEL_DTYPES:
        return True

    if x.size(-1) > _MAX_KERNEL_BLOCK_D:
        return True

    return False


def _should_use_eager_mipe(
    *,
    position_ids: torch.Tensor,
    window: torch.Tensor,
) -> bool:
    if position_ids.numel() == 0 or window.numel() == 0:
        return True

    if not position_ids.is_cuda:
        return True

    if window.dtype not in _SUPPORTED_KERNEL_DTYPES:
        return True

    if torch.is_floating_point(position_ids) and (
        position_ids.dtype not in _SUPPORTED_KERNEL_DTYPES
    ):
        return True

    if torch.result_type(position_ids, window) not in _SUPPORTED_KERNEL_DTYPES:
        return True

    return False


def _should_use_eager_apply_mipe(
    *,
    sequence: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> bool:
    if sequence.numel() == 0:
        return True

    if not sequence.is_cuda:
        return True

    if sequence.dtype not in _SUPPORTED_KERNEL_DTYPES:
        return True

    if freqs_cis.dtype not in _SUPPORTED_KERNEL_DTYPES:
        return True

    if torch.result_type(sequence, freqs_cis) not in _SUPPORTED_KERNEL_DTYPES:
        return True

    return False


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
