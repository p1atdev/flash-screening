import pytest
import torch

from flash_screening import (
    apply_mipe,
    compute_freqs_cis,
    flash_screening,
    mipe_rotation,
    unit_length_norm,
)
from flash_screening.eager import (
    apply_mipe as eager_apply_mipe,
    compute_freqs_cis as eager_compute_freqs_cis,
    mipe_rotation as eager_mipe_rotation,
    screening as eager_screening,
    unit_length_norm as eager_unit_length_norm,
)


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


def _clone_requires_grad(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().requires_grad_()


def _run_backward(
    *,
    use_flash: bool,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    acceptance: torch.Tensor,
    window: torch.Tensor,
    grad_output: torch.Tensor,
    position_ids: torch.Tensor | None = None,
    attention_mask: torch.Tensor | None = None,
    is_causal: bool = True,
) -> tuple[torch.Tensor, tuple[torch.Tensor | None, ...]]:
    query = _clone_requires_grad(query)
    key = _clone_requires_grad(key)
    value = _clone_requires_grad(value)
    acceptance = _clone_requires_grad(acceptance)
    window = _clone_requires_grad(window)
    fn = flash_screening if use_flash else eager_screening

    output = fn(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        position_ids=position_ids,
        attention_mask=attention_mask,
        is_causal=is_causal,
    )
    assert isinstance(output, torch.Tensor)
    (output * grad_output).sum().backward()

    grads = tuple(
        tensor.grad.detach() if tensor.grad is not None else None
        for tensor in (query, key, value, acceptance, window)
    )
    return output.detach(), grads


def _assert_grads_close(
    actual: tuple[torch.Tensor | None, ...],
    expected: tuple[torch.Tensor | None, ...],
) -> None:
    for actual_grad, expected_grad in zip(actual, expected, strict=True):
        if expected_grad is None:
            assert actual_grad is None
        else:
            assert actual_grad is not None
            torch.testing.assert_close(actual_grad, expected_grad, rtol=6e-2, atol=6e-2)


def test_mipe_helpers_cpu_fall_back_to_eager() -> None:
    position_ids = torch.arange(6, dtype=torch.float32)[None, :].expand(2, -1)
    window = torch.tensor([3.5, 128.0, 300.0])
    sequence = torch.randn(2, 3, 6, 7)

    torch.testing.assert_close(
        unit_length_norm(sequence),
        eager_unit_length_norm(sequence),
    )
    torch.testing.assert_close(
        mipe_rotation(position_ids, window),
        eager_mipe_rotation(position_ids, window),
    )

    position_ids_3d = torch.stack((position_ids, position_ids * 2.0), dim=-1)
    freqs = compute_freqs_cis(position_ids_3d, window)
    expected_freqs = eager_compute_freqs_cis(position_ids_3d, window)
    torch.testing.assert_close(freqs, expected_freqs)
    torch.testing.assert_close(
        apply_mipe(sequence, freqs),
        eager_apply_mipe(sequence, expected_freqs),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_unit_length_norm_cuda_backward_matches_eager() -> None:
    torch.manual_seed(1)
    x = torch.randn(2, 3, 13, device="cuda")
    grad_output = torch.randn_like(x)

    actual_x = _clone_requires_grad(x)
    expected_x = _clone_requires_grad(x)

    actual = unit_length_norm(actual_x, eps=1e-5)
    expected = eager_unit_length_norm(expected_x, eps=1e-5)
    (actual * grad_output).sum().backward()
    (expected * grad_output).sum().backward()

    _assert_close(actual, expected)
    torch.testing.assert_close(actual_x.grad, expected_x.grad, rtol=6e-2, atol=6e-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mipe_rotation_cuda_backward_matches_eager() -> None:
    base = torch.arange(7, device="cuda", dtype=torch.float32)
    position_ids = torch.stack((base, base * 1.5 + 0.25), dim=0)
    window = torch.tensor([3.5, 128.0, 300.0], device="cuda")

    actual_position_ids = _clone_requires_grad(position_ids)
    expected_position_ids = _clone_requires_grad(position_ids)
    actual_window = _clone_requires_grad(window)
    expected_window = _clone_requires_grad(window)

    actual = mipe_rotation(actual_position_ids, actual_window)
    expected = eager_mipe_rotation(expected_position_ids, expected_window)
    grad_output = torch.randn_like(actual)
    (actual * grad_output).sum().backward()
    (expected * grad_output).sum().backward()

    _assert_close(actual, expected)
    torch.testing.assert_close(
        actual_position_ids.grad,
        expected_position_ids.grad,
        rtol=6e-2,
        atol=6e-2,
    )
    torch.testing.assert_close(
        actual_window.grad,
        expected_window.grad,
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compute_freqs_cis_cuda_backward_matches_eager() -> None:
    base = torch.arange(6, device="cuda", dtype=torch.float32)
    position_ids = torch.stack((base, base * 2.0 + 1.0), dim=-1)
    position_ids = position_ids[None, :, :].expand(2, -1, -1).contiguous()
    window = torch.tensor([4.0, 96.0, 280.0], device="cuda")

    actual_position_ids = _clone_requires_grad(position_ids)
    expected_position_ids = _clone_requires_grad(position_ids)
    actual_window = _clone_requires_grad(window)
    expected_window = _clone_requires_grad(window)

    actual = compute_freqs_cis(actual_position_ids, actual_window)
    expected = eager_compute_freqs_cis(expected_position_ids, expected_window)
    grad_output = torch.randn_like(actual)
    (actual * grad_output).sum().backward()
    (expected * grad_output).sum().backward()

    _assert_close(actual, expected)
    torch.testing.assert_close(
        actual_position_ids.grad,
        expected_position_ids.grad,
        rtol=6e-2,
        atol=6e-2,
    )
    torch.testing.assert_close(
        actual_window.grad,
        expected_window.grad,
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_apply_mipe_cuda_backward_matches_eager() -> None:
    torch.manual_seed(2)
    sequence = torch.randn(2, 3, 5, 7, device="cuda")
    freqs = torch.randn(2, 3, 5, 4, device="cuda")

    actual_sequence = _clone_requires_grad(sequence)
    expected_sequence = _clone_requires_grad(sequence)
    actual_freqs = _clone_requires_grad(freqs)
    expected_freqs = _clone_requires_grad(freqs)

    actual = apply_mipe(actual_sequence, actual_freqs)
    expected = eager_apply_mipe(expected_sequence, expected_freqs)
    grad_output = torch.randn_like(actual)
    (actual * grad_output).sum().backward()
    (expected * grad_output).sum().backward()

    _assert_close(actual, expected)
    torch.testing.assert_close(
        actual_sequence.grad,
        expected_sequence.grad,
        rtol=6e-2,
        atol=6e-2,
    )
    torch.testing.assert_close(
        actual_freqs.grad,
        expected_freqs.grad,
        rtol=6e-2,
        atol=6e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_mipe_pipeline_cuda_backward_matches_eager() -> None:
    torch.manual_seed(3)
    sequence = torch.randn(2, 3, 6, 6, device="cuda")
    base = torch.arange(6, device="cuda", dtype=torch.float32)
    position_ids = torch.stack((base, base * 0.5 + 2.0), dim=-1)
    position_ids = position_ids[None, :, :].expand(2, -1, -1).contiguous()
    window = torch.tensor([8.0, 128.0, 320.0], device="cuda")
    grad_output = torch.randn_like(sequence)

    actual_sequence = _clone_requires_grad(sequence)
    expected_sequence = _clone_requires_grad(sequence)
    actual_position_ids = _clone_requires_grad(position_ids)
    expected_position_ids = _clone_requires_grad(position_ids)
    actual_window = _clone_requires_grad(window)
    expected_window = _clone_requires_grad(window)

    actual = apply_mipe(
        unit_length_norm(actual_sequence),
        compute_freqs_cis(actual_position_ids, actual_window),
    )
    expected = eager_apply_mipe(
        eager_unit_length_norm(expected_sequence),
        eager_compute_freqs_cis(expected_position_ids, expected_window),
    )
    (actual * grad_output).sum().backward()
    (expected * grad_output).sum().backward()

    _assert_close(actual, expected)
    for actual_grad, expected_grad in (
        (actual_sequence.grad, expected_sequence.grad),
        (actual_position_ids.grad, expected_position_ids.grad),
        (actual_window.grad, expected_window.grad),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=6e-2, atol=6e-2)


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
def test_flash_screening_cuda_backward_matches_eager() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=11)
    grad_output = torch.randn_like(value)
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=query.dtype,
    )
    attention_mask[:, ::4] = 0

    actual, actual_grads = _run_backward(
        use_flash=True,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        attention_mask=attention_mask,
    )
    expected, expected_grads = _run_backward(
        use_flash=False,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        attention_mask=attention_mask,
    )

    _assert_close(actual, expected)
    _assert_grads_close(actual_grads, expected_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_cuda_backward_matches_eager_with_position_ids() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=8)
    grad_output = torch.randn_like(value)
    position_ids = torch.arange(query.size(2), device=query.device)[None, :, None]
    position_ids = position_ids.expand(query.size(0), -1, -1).contiguous()
    position_ids[1, :, 0] *= 2
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=query.dtype,
    )
    attention_mask[:, ::3] = 0

    actual, actual_grads = _run_backward(
        use_flash=True,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )
    expected, expected_grads = _run_backward(
        use_flash=False,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    _assert_close(actual, expected)
    _assert_grads_close(actual_grads, expected_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_cuda_backward_matches_eager_non_causal() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=8)
    grad_output = torch.randn_like(value)
    position_ids = torch.stack(
        [
            torch.arange(query.size(2), device=query.device),
            torch.arange(query.size(2), device=query.device) * 2,
        ],
        dim=-1,
    )[None, :, :].expand(query.size(0), -1, -1)

    actual, actual_grads = _run_backward(
        use_flash=True,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        position_ids=position_ids,
        is_causal=False,
    )
    expected, expected_grads = _run_backward(
        use_flash=False,
        query=query,
        key=key,
        value=value,
        acceptance=acceptance,
        window=window,
        grad_output=grad_output,
        position_ids=position_ids,
        is_causal=False,
    )

    _assert_close(actual, expected)
    _assert_grads_close(actual_grads, expected_grads)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_non_causal_unused_window_does_not_require_grad() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=8)
    window = window.requires_grad_()

    actual = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        is_causal=False,
    )

    assert not actual.requires_grad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_attention_mask_grad_uses_eager() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=8)
    attention_mask = torch.ones(
        query.size(0),
        query.size(2),
        device=query.device,
        dtype=query.dtype,
        requires_grad=True,
    )

    actual = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        attention_mask=attention_mask,
    )
    actual.sum().backward()

    assert attention_mask.grad is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flash_screening_return_score_grad_uses_eager() -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=8)
    query.requires_grad_(True)

    actual, score = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
        return_score=True,
    )
    (actual.sum() + score.sum()).backward()

    assert query.grad is not None


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
    position_ids = torch.stack(
        [
            torch.arange(query.size(2), device=query.device),
            torch.arange(query.size(2), device=query.device) * 2,
        ],
        dim=-1,
    )[None, :, :].expand(query.size(0), -1, -1)
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
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_causal=False,
        )
        expected = eager_screening(
            query,
            key,
            value,
            acceptance=acceptance,
            window=window,
            position_ids=position_ids,
            attention_mask=attention_mask,
            is_causal=False,
        )

    assert isinstance(expected, torch.Tensor)
    _assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flash_screening_cuda_backward_low_precision_smoke(dtype: torch.dtype) -> None:
    query, key, value, acceptance, window = _inputs(device="cuda", seq_len=5)
    query = query.to(dtype).requires_grad_()
    key = key.to(dtype).requires_grad_()
    value = value.to(dtype).requires_grad_()
    acceptance = acceptance.to(dtype).requires_grad_()
    window = window.to(dtype).requires_grad_()

    actual = flash_screening(
        query,
        key,
        value,
        acceptance=acceptance,
        window=window,
    )
    assert actual.dtype == dtype
    actual.sum().backward()

    assert actual.requires_grad
    assert query.grad is not None
    assert key.grad is not None
    assert value.grad is not None
    assert acceptance.grad is not None
    assert window.grad is not None
