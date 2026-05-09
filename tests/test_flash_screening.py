import pytest
import torch

from flash_screening import flash_screening
from flash_screening.eager import screening as eager_screening


def _inputs(
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    batch_size: int = 2,
    num_heads: int = 3,
    seq_len: int = 17,
    key_dim: int = 16,
    value_dim: int = 12,
) -> tuple[torch.Tensor, ...]:
    torch.manual_seed(0)
    query = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        key_dim,
        device=device,
        dtype=dtype,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        batch_size,
        num_heads,
        seq_len,
        value_dim,
        device=device,
        dtype=dtype,
    )
    acceptance = torch.linspace(
        0.35,
        0.95,
        num_heads,
        device=device,
        dtype=dtype,
    )
    window = torch.tensor([3.5, 7.25, 32.0], device=device, dtype=dtype)
    return query, key, value, acceptance, window


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)


def _assert_low_precision_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=8e-2,
        atol=8e-2,
    )


def test_flash_screening_cpu_falls_back_to_eager() -> None:
    query, key, value, acceptance, window = _inputs(device="cpu", seq_len=9)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 1, 1, 0, 1, 1],
            [1, 0, 1, 1, 1, 0, 1, 1, 0],
        ],
        dtype=query.dtype,
    )

    actual, actual_score = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        attention_mask=attention_mask,
        return_score=True,
    )
    expected, expected_score = eager_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        attention_mask=attention_mask,
        return_score=True,
    )

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_score, expected_score)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_screening_cuda_matches_eager_low_precision(dtype: torch.dtype) -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", dtype=dtype)
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=dtype,
    )
    attention_mask[:, ::5] = 0

    with torch.no_grad():
        actual, actual_score = flash_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            attention_mask=attention_mask,
            return_score=True,
        )
        expected, expected_score = eager_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            attention_mask=attention_mask,
            return_score=True,
        )

    assert actual.dtype == dtype
    assert actual_score.dtype == dtype
    _assert_low_precision_close(actual, expected)
    _assert_low_precision_close(actual_score, expected_score)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_cuda_matches_eager_default_positions() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda")

    with torch.no_grad():
        actual, actual_score = flash_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            return_score=True,
        )
        expected, expected_score = eager_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            return_score=True,
        )

    _assert_close(actual, expected)
    _assert_close(actual_score, expected_score)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_cuda_matches_eager_with_position_ids_and_mask() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda")
    position_ids = torch.arange(query.size(2), device=query.device)[None, :, None]
    position_ids = position_ids.expand(query.size(0), -1, -1).contiguous()
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=query.dtype,
    )
    attention_mask[:, ::4] = 0

    with torch.no_grad():
        actual = flash_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        expected = eager_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

    assert isinstance(expected, torch.Tensor)
    _assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_cuda_matches_eager_non_causal() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda")
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=query.dtype,
    )
    attention_mask[:, 1::3] = 0

    with torch.no_grad():
        actual = flash_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            attention_mask=attention_mask,
            is_causal=False,
        )
        expected = eager_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            attention_mask=attention_mask,
            is_causal=False,
        )

    assert isinstance(expected, torch.Tensor)
    _assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_uses_eager_when_gradients_are_required() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=5)
    query.requires_grad_(True)

    actual = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )

    assert actual.requires_grad
