# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import os

import torch

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

import warnings

__all__ = [
    'flash_attention',
    'sdpa_attention',
    'attention',
]

# auto|cudnn|flash|sdpa. auto picks cuDNN when it is equivalent (~2x faster on
# sm_100, slower on Hopper), so it is resolved per-device rather than pinned.
ATTN_BACKEND = os.getenv('WAN_ATTN_BACKEND', 'auto').lower()

_CUDNN_SDPA_OK = None
_WARNED = set()


def _warn_once(key, msg):
    if key not in _WARNED:
        _WARNED.add(key)
        warnings.warn(msg)


def _flash_available():
    return FLASH_ATTN_2_AVAILABLE or FLASH_ATTN_3_AVAILABLE


def _cudnn_sdpa_ok():
    """Probe the cuDNN SDPA backend once; not all builds/shapes support it."""
    global _CUDNN_SDPA_OK
    if _CUDNN_SDPA_OK is None:
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            x = torch.zeros(1, 2, 8, 64, device='cuda', dtype=torch.bfloat16)
            with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
                torch.nn.functional.scaled_dot_product_attention(x, x, x)
            _CUDNN_SDPA_OK = True
        except Exception:
            _CUDNN_SDPA_OK = False
    return _CUDNN_SDPA_OK


def _unpadded(lens, length):
    """True when `lens` imposes no mask, so SDPA is equivalent to varlen."""
    return lens is None or int(lens.min()) >= length


def flash_attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    version=None,
):
    """
    q:              [B, Lq, Nq, C1].
    k:              [B, Lk, Nk, C1].
    v:              [B, Lk, Nk, C2]. Nq must be divisible by Nk.
    q_lens:         [B].
    k_lens:         [B].
    dropout_p:      float. Dropout probability.
    softmax_scale:  float. The scaling of QK^T before applying softmax.
    causal:         bool. Whether to apply causal attention mask.
    window_size:    (left right). If not (-1, -1), apply sliding window local attention.
    deterministic:  bool. If True, slightly slower and uses more memory.
    dtype:          torch.dtype. Apply when dtype of q/k/v is not float16/bfloat16.
    """
    half_dtypes = (torch.float16, torch.bfloat16)
    assert dtype in half_dtypes
    assert q.device.type == 'cuda' and q.size(-1) <= 256

    # params
    b, lq, lk, out_dtype = q.size(0), q.size(1), k.size(1), q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    # preprocess query
    if q_lens is None:
        q = half(q.flatten(0, 1))
        q_lens = torch.tensor(
            [lq] * b, dtype=torch.int32).to(
                device=q.device, non_blocking=True)
    else:
        q = half(torch.cat([u[:v] for u, v in zip(q, q_lens)]))

    # preprocess key, value
    if k_lens is None:
        k = half(k.flatten(0, 1))
        v = half(v.flatten(0, 1))
        k_lens = torch.tensor(
            [lk] * b, dtype=torch.int32).to(
                device=k.device, non_blocking=True)
    else:
        k = half(torch.cat([u[:v] for u, v in zip(k, k_lens)]))
        v = half(torch.cat([u[:v] for u, v in zip(v, k_lens)]))

    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    if version is not None and version == 3 and not FLASH_ATTN_3_AVAILABLE:
        warnings.warn(
            'Flash attention 3 is not available, use flash attention 2 instead.'
        )

    # apply attention
    if (version is None or version == 3) and FLASH_ATTN_3_AVAILABLE:
        # Note: dropout_p, window_size are not supported in FA3 now.
        x = flash_attn_interface.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            seqused_q=None,
            seqused_k=None,
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            softmax_scale=softmax_scale,
            causal=causal,
            deterministic=deterministic)[0]
    else:
        assert FLASH_ATTN_2_AVAILABLE
        x = flash_attn.flash_attn_varlen_func(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=torch.cat([q_lens.new_zeros([1]), q_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            cu_seqlens_k=torch.cat([k_lens.new_zeros([1]), k_lens]).cumsum(
                0, dtype=torch.int32).to(q.device, non_blocking=True),
            max_seqlen_q=lq,
            max_seqlen_k=lk,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic)

    # varlen returns sum(q_lens) rows; that is b*lq only when q_lens is uniform.
    if x.size(0) == b * lq:
        x = x.unflatten(0, (b, lq))
    else:
        packed, x = x, x.new_zeros(b, lq, *x.shape[1:])
        offset = 0
        for i, n in enumerate(q_lens.tolist()):
            x[i, :n] = packed[offset:offset + n]
            offset += n

    # output
    return x.type(out_dtype)


def sdpa_attention(
    q,
    k,
    v,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    dtype=torch.bfloat16,
    force_cudnn=False,
):
    """torch SDPA over [B, L, N, C] inputs; cannot mask padding."""
    half_dtypes = (torch.float16, torch.bfloat16)
    out_dtype = q.dtype

    def half(x):
        return x if x.dtype in half_dtypes else x.to(dtype)

    q, k, v = half(q), half(k), half(v)
    q = q.to(v.dtype)
    k = k.to(v.dtype)

    if q_scale is not None:
        q = q * q_scale

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    def _run():
        return torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            is_causal=causal,
            dropout_p=dropout_p,
            scale=softmax_scale)

    if force_cudnn:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            out = _run()
    else:
        out = _run()

    return out.transpose(1, 2).contiguous().type(out_dtype)


def attention(
    q,
    k,
    v,
    q_lens=None,
    k_lens=None,
    dropout_p=0.,
    softmax_scale=None,
    q_scale=None,
    causal=False,
    window_size=(-1, -1),
    deterministic=False,
    dtype=torch.bfloat16,
    fa_version=None,
):
    """Dispatch to the fastest backend that is correct here; padding and sliding
    windows always route to flash-attn, whatever the preference."""
    windowed = tuple(window_size) != (-1, -1)
    unpadded = _unpadded(q_lens, q.size(1)) and _unpadded(k_lens, k.size(1))
    needs_flash = windowed or not unpadded

    backend = ATTN_BACKEND
    if backend not in ('auto', 'cudnn', 'flash', 'sdpa'):
        _warn_once('bad-backend',
                   f'Unknown WAN_ATTN_BACKEND={backend!r}, using auto.')
        backend = 'auto'

    if needs_flash:
        if _flash_available():
            if backend in ('cudnn', 'sdpa'):
                _warn_once(
                    'needs-flash', f'WAN_ATTN_BACKEND={backend} cannot express '
                    f'{"a sliding window" if windowed else "padded q/k lens"}; '
                    'using flash attention for those calls to stay correct.')
            backend = 'flash'
        else:
            # SDPA is all that is left, and it will attend over padding.
            _warn_once(
                'sdpa-padded',
                'Padding mask is disabled when using scaled_dot_product_'
                'attention. It can have a significant impact on performance.')
            backend = 'sdpa'
    elif backend == 'auto':
        backend = 'cudnn' if (q.is_cuda and _cudnn_sdpa_ok()) else (
            'flash' if _flash_available() else 'sdpa')
    elif backend == 'flash' and not _flash_available():
        _warn_once(
            'no-flash',
            'WAN_ATTN_BACKEND=flash but flash-attn is not installed; '
            'falling back to torch SDPA.')
        backend = 'sdpa'

    if backend == 'flash':
        return flash_attention(
            q=q,
            k=k,
            v=v,
            q_lens=q_lens,
            k_lens=k_lens,
            dropout_p=dropout_p,
            softmax_scale=softmax_scale,
            q_scale=q_scale,
            causal=causal,
            window_size=window_size,
            deterministic=deterministic,
            dtype=dtype,
            version=fa_version,
        )

    return sdpa_attention(
        q=q,
        k=k,
        v=v,
        dropout_p=dropout_p,
        softmax_scale=softmax_scale,
        q_scale=q_scale,
        causal=causal,
        dtype=dtype,
        force_cudnn=(backend == 'cudnn'),
    )
