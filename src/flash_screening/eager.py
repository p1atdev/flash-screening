import torch


def unit_length_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return x / x.norm(p=2, dim=-1, keepdim=True).clip(min=eps)


# MiPE is a RoPE-like rotation [18] applied to one feature pair per position
# axis, with a rotation angle modulated by the learned screening window w.
def mipe_rotation(
    position_ids: torch.Tensor,  # [batch_size, seq_len]
    window: torch.Tensor,  # [num_heads]
    window_threshold: float = 256.0,
) -> torch.Tensor:
    batch_size, seq_len = position_ids.size()
    num_heads = window.size(0)
    window = window[None, :, None].repeat(
        batch_size, 1, seq_len
    )  # [batch_size, num_heads, seq_len]

    # gamma(w)
    gamma = torch.where(
        window < window_threshold,
        (torch.cos(torch.pi * window / window_threshold) + 1) / 2,
        torch.zeros_like(window),
    )

    position_ids = position_ids[:, None, :].repeat(
        1, num_heads, 1
    )  # [batch_size, num_heads, seq_len]

    rotation = torch.pi * position_ids * gamma / window

    return rotation


def compute_freqs_cis(
    position_ids: torch.Tensor,  # [batch_size, seq_len, num_axes]
    window: torch.Tensor,  # [num_heads]
    window_threshold: float = 256.0,
) -> torch.Tensor:
    freqs_cis = []  # [batch_size, num_heads, seq_len, 2*num_axes]

    if position_ids.ndim == 2:
        position_ids = position_ids.unsqueeze(-1)  # [batch_size, seq_len, 1]

    for axis in range(position_ids.size(-1)):
        rotation = mipe_rotation(
            position_ids=position_ids[..., axis],
            window=window,
            window_threshold=window_threshold,
        )  # [batch_size, num_heads, seq_len]

        freqs_cis.append(
            torch.stack([torch.cos(rotation), torch.sin(rotation)], dim=-1)
        )
        # [batch_size, num_heads, seq_len, 2]

    return torch.cat(freqs_cis, dim=-1).float()


def apply_mipe(
    sequence: torch.Tensor,  # [batch_size, num_heads, seq_len, head_dim]
    freqs_cis: torch.Tensor,  # [batch_size, num_heads, seq_len, 2 (cos, sin) * num_axes]
) -> torch.Tensor:
    _, _, _, encoded_dim = freqs_cis.size()

    assert encoded_dim % 2 == 0, "encoded_dim must be even"
    assert sequence.size(-1) >= encoded_dim, "head_dim must be >= encoded_dim"

    seq = sequence.float()

    x_even = seq[..., :encoded_dim:2]
    x_odd = seq[..., 1:encoded_dim:2]

    cos = freqs_cis[..., :encoded_dim:2]
    sin = freqs_cis[..., 1:encoded_dim:2]

    x1 = x_even * cos - x_odd * sin
    x2 = x_even * sin + x_odd * cos

    rotated = torch.stack((x1, x2), dim=-1).flatten(-2)

    return torch.cat(
        [rotated, seq[..., encoded_dim:]],
        dim=-1,
    ).type_as(sequence)


def trim_similarity(
    similarity: torch.Tensor,  # [batch_size, num_heads, seq_len, seq_len]
    acceptance: torch.Tensor,  # [num_heads]
) -> torch.Tensor:

    relevance = (
        torch.max(
            1 - (1 - similarity) / acceptance[None, :, None, None],
            torch.zeros_like(similarity),
        )
        ** 2
    )

    return relevance


def tanh_norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    norm = x.norm(p=2, dim=-1, keepdim=True)
    scale = torch.where(
        norm > eps,
        torch.tanh(norm) / norm.clamp_min(eps),
        torch.ones_like(norm),
    )
    return scale * x


def causal_softmask(
    position_ids: torch.Tensor,  # [batch_size, seq_len] or [batch_size, seq_len, 1]
    window: torch.Tensor,  # [num_heads]
) -> torch.Tensor:
    if position_ids.ndim == 3 and position_ids.size(-1) == 1:
        position_ids = position_ids.squeeze(-1)

    assert position_ids.ndim == 2, (
        "Causal softmask only supports position ids shaped [batch_size, seq_len] "
        "or [batch_size, seq_len, 1]"
    )

    position_diff = position_ids[:, None, :] - position_ids[:, :, None]
    # [batch_size, seq_len, seq_len]
    position_diff = position_diff[:, None, :, :].repeat(
        1, window.size(0), 1, 1
    )  # [batch_size, num_heads, seq_len, seq_len]

    window = window[None, :, None, None]

    mask = torch.where(
        (-window < position_diff) & (position_diff <= 0),
        (torch.cos(torch.pi * position_diff / window) + 1) / 2,
        torch.zeros_like(position_diff),
    )

    return mask


def screening(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    #
    acceptance: torch.Tensor,  # similarity acceptance
    window: torch.Tensor,
    #
    position_ids: torch.Tensor | None = None,  # [batch_size, seq_len, num_axes]
    attention_mask: torch.Tensor | None = None,  # [batch_size, seq_len] mask
    is_causal: bool = True,
    return_score: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:

    similarity = query @ key.transpose(-2, -1)

    # Trim
    relevance = trim_similarity(similarity, acceptance)

    # Softmask
    if is_causal:
        softmask_position_ids = position_ids
        if softmask_position_ids is None:
            batch_size, _, seq_len, _ = query.size()
            softmask_position_ids = torch.arange(
                seq_len,
                device=query.device,
            )[None, :].expand(batch_size, -1)

        softmask = causal_softmask(
            position_ids=softmask_position_ids,
            window=window,
        )  # [batch_size, num_heads, seq_len, seq_len]
        softmask = softmask.to(dtype=relevance.dtype)
        relevance = relevance * softmask

    # Optional attention mask (e.g., for padding tokens)
    if attention_mask is not None:
        mask = attention_mask.to(device=relevance.device, dtype=relevance.dtype)
        relevance = relevance * mask[:, None, None, :]

    # @
    screened = relevance @ value

    # TanhNorm
    screened = tanh_norm(screened)

    if return_score:
        return screened, relevance

    return screened
