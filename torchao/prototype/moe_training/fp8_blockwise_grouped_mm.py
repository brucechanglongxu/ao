# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
FP8 blockwise grouped matrix multiplication for MoE training.

Uses (1, block_size) scaling granularity for activations and
(block_size, block_size) scaling for weights, following the DeepSeek-V3
blockwise FP8 approach adapted for grouped GEMM.

Two execution paths:

1. **Triton path** (use_triton=True, default): blockwise-quantizes to FP8,
   then runs a custom Triton GEMM kernel per expert that fuses dequantization
   into the accumulation loop. The full dequantized tensors are never
   materialized, giving real FP8 compute savings.

2. **Emulated path** (use_triton=False): blockwise-quantizes to FP8, then
   dequantizes back to high precision before performing the grouped GEMM via
   torch._grouped_mm. Useful for correctness validation on any hardware.
"""

from typing import Optional

import torch
import torch.nn.functional as F

from torchao.prototype.moe_training.kernels.float8_blockwise import (
    fp8_blockwise_gemm,
)


def _to_fp8_blockwise_then_scaled_grouped_mm(
    A: torch.Tensor,
    B_t: torch.Tensor,
    offs: torch.Tensor,
    block_size: int = 128,
    out_dtype: Optional[torch.dtype] = torch.bfloat16,
    use_triton: bool = True,
) -> torch.Tensor:
    """
    Differentiable FP8 grouped matrix multiplication with blockwise quantization.

    Args:
        A: Left operand, shape (M, K), row-major. K must be divisible by block_size.
        B_t: Right operand, shape (E, K, N). K and N must be divisible by block_size.
        offs: Group boundary offsets, shape (num_groups,), int32.
        block_size: Block size for quantization (default 128).
        out_dtype: Output dtype (default bfloat16).
        use_triton: If True, use Triton GEMM kernel with fused dequant.
            If False, use emulated path.

    Returns:
        Result of grouped matmul, shape (M, N).
    """
    return _Float8BlockwiseGroupedMM.apply(
        A, B_t, offs, block_size, out_dtype, use_triton
    )


# ---------------------------------------------------------------------------
# Blockwise FP8 quantization helpers (pure PyTorch, platform-agnostic)
# ---------------------------------------------------------------------------


def _fp8_blockwise_act_quant(
    x: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize activation tensor with (1, block_size) granularity.

    Returns (fp8_data, scales) where scales = amax / fp8_max (the value
    needed to *multiply* by to dequantize).
    """
    assert x.is_contiguous(), "Input must be contiguous"
    assert x.size(-1) % block_size == 0

    orig_shape = x.shape
    x_flat = x.reshape(-1, x.size(-1))
    M, K = x_flat.shape
    num_blocks = K // block_size

    x_blocked = x_flat.reshape(M, num_blocks, block_size)
    amax = x_blocked.abs().amax(dim=-1)  # (M, num_blocks)
    scales = amax / torch.finfo(torch.float8_e4m3fn).max
    scales = scales.clamp(min=1e-12)

    x_scaled = x_blocked / scales.unsqueeze(-1)
    fp8_data = x_scaled.clamp(
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)

    fp8_data = fp8_data.reshape(orig_shape)
    scales = scales.reshape(*orig_shape[:-1], num_blocks)
    return fp8_data, scales


def _fp8_blockwise_weight_quant(
    w: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize weight tensor with (block_size, block_size) granularity.

    Returns (fp8_data, scales) where scales = amax / fp8_max.
    """
    assert w.ndim == 2
    K, N = w.shape
    assert K % block_size == 0 and N % block_size == 0

    num_k_blocks = K // block_size
    num_n_blocks = N // block_size

    w_blocked = w.reshape(num_k_blocks, block_size, num_n_blocks, block_size)
    w_blocked = w_blocked.permute(0, 2, 1, 3)  # (num_k, num_n, bs, bs)
    amax = w_blocked.abs().amax(dim=(-2, -1))  # (num_k, num_n)
    scales = amax / torch.finfo(torch.float8_e4m3fn).max
    scales = scales.clamp(min=1e-12)

    w_scaled = w_blocked / scales.unsqueeze(-1).unsqueeze(-1)
    fp8_data = w_scaled.clamp(
        min=torch.finfo(torch.float8_e4m3fn).min,
        max=torch.finfo(torch.float8_e4m3fn).max,
    ).to(torch.float8_e4m3fn)

    fp8_data = fp8_data.permute(0, 2, 1, 3).reshape(K, N)
    return fp8_data, scales


def _fp8_blockwise_dequant_act(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize activation from FP8 with (1, block_size) scales."""
    orig_shape = fp8_data.shape
    K = orig_shape[-1]
    num_blocks = K // block_size

    data = fp8_data.reshape(*orig_shape[:-1], num_blocks, block_size).to(out_dtype)
    data = data * scales.unsqueeze(-1).to(out_dtype)
    return data.reshape(orig_shape)


def _fp8_blockwise_dequant_weight(
    fp8_data: torch.Tensor,
    scales: torch.Tensor,
    block_size: int,
    out_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Dequantize weight from FP8 with (block_size, block_size) scales."""
    K, N = fp8_data.shape
    num_k_blocks = K // block_size
    num_n_blocks = N // block_size

    data = fp8_data.reshape(num_k_blocks, block_size, num_n_blocks, block_size)
    data = data.permute(0, 2, 1, 3).to(out_dtype)  # (num_k, num_n, bs, bs)
    data = data * scales.unsqueeze(-1).unsqueeze(-1).to(out_dtype)
    data = data.permute(0, 2, 1, 3).reshape(K, N)
    return data


# ---------------------------------------------------------------------------
# Autograd function
# ---------------------------------------------------------------------------


class _Float8BlockwiseGroupedMM(torch.autograd.Function):
    """
    Differentiable grouped GEMM with blockwise FP8 quantization.

    Forward: A (M,K) @ B_t (E,K,N) -> (M,N) with group offsets.
    Backward: computes grad_A and grad_B using the same blockwise quantization.
    """

    @staticmethod
    def forward(
        ctx,
        A: torch.Tensor,
        B_t: torch.Tensor,
        offs: Optional[torch.Tensor] = None,
        block_size: int = 128,
        out_dtype: Optional[torch.dtype] = torch.bfloat16,
        use_triton: bool = True,
    ) -> torch.Tensor:
        assert A.ndim == 2, "A must be 2D"
        assert B_t.ndim == 3, "B_t must be 3D"
        assert A.size(-1) == B_t.size(-2), (
            f"Incompatible shapes: A={A.shape}, B_t={B_t.shape}"
        )
        assert A.size(-1) % block_size == 0
        assert B_t.size(-2) % block_size == 0 and B_t.size(-1) % block_size == 0

        ctx.save_for_backward(A, B_t, offs)
        ctx.block_size = block_size
        ctx.out_dtype = out_dtype
        ctx.use_triton = use_triton

        if use_triton:
            return _forward_triton(A, B_t, offs, block_size, out_dtype)
        else:
            return _forward_emulated(A, B_t, offs, block_size, out_dtype)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        A, B_t, offs = ctx.saved_tensors
        block_size = ctx.block_size
        out_dtype = ctx.out_dtype
        use_triton = ctx.use_triton

        if use_triton:
            grad_A, grad_B = _backward_triton(
                grad_output, A, B_t, offs, block_size, out_dtype
            )
        else:
            grad_A, grad_B = _backward_emulated(
                grad_output, A, B_t, offs, block_size, out_dtype
            )

        return grad_A, grad_B, None, None, None, None


# ---------------------------------------------------------------------------
# Triton path: per-expert loop with fused FP8 GEMM kernel
# ---------------------------------------------------------------------------


def _forward_triton(A, B_t, offs, block_size, out_dtype):
    """Forward using Triton FP8 GEMM per expert."""
    E, K, N = B_t.shape
    M = A.shape[0]
    output = torch.zeros(M, N, dtype=out_dtype, device=A.device)

    # Quantize A once (shared across all experts)
    A_fp8, A_scales = _fp8_blockwise_act_quant(A.contiguous(), block_size)

    start = 0
    for e in range(E):
        end = offs[e].item()
        M_e = end - start
        if M_e == 0:
            start = end
            continue

        # Slice quantized A for this expert
        A_e_fp8 = A_fp8[start:end].contiguous()
        A_e_scales = A_scales[start:end].contiguous()

        # Quantize this expert's weight: (K, N) -> FP8 with (bs, bs) scales
        W_e = B_t[e].contiguous()  # (K, N)
        W_e_fp8, W_e_scales = _fp8_blockwise_weight_quant(W_e, block_size)

        # Triton GEMM: (M_e, K) @ (K, N) -> (M_e, N)
        output[start:end] = fp8_blockwise_gemm(
            A_e_fp8, W_e_fp8, A_e_scales, W_e_scales, block_size, out_dtype
        )

        start = end
    return output


def _backward_triton(grad_output, A, B_t, offs, block_size, out_dtype):
    """Backward using Triton FP8 GEMM per expert."""
    E, K, N = B_t.shape
    M = A.shape[0]
    B_hp = B_t._data if hasattr(B_t, "_data") else B_t
    A_hp = A._data if hasattr(A, "_data") else A

    grad_A = torch.zeros(M, K, dtype=out_dtype, device=A.device)
    grad_B = torch.zeros(E, K, N, dtype=out_dtype, device=A.device)

    # Quantize grad_output once for grad_A computation
    grad_fp8, grad_scales = _fp8_blockwise_act_quant(
        grad_output.contiguous(), block_size
    )

    start = 0
    for e in range(E):
        end = offs[e].item()
        M_e = end - start
        if M_e == 0:
            start = end
            continue

        # --- grad_A_e = grad_output_e @ B_e ---
        # grad_output_e: (M_e, N), B_e = B_t[e]^T = (N, K)
        # GEMM: (M_e, N) @ (N, K) -> (M_e, K)
        grad_e_fp8 = grad_fp8[start:end].contiguous()
        grad_e_scales = grad_scales[start:end].contiguous()

        B_e_T = B_hp[e].t().contiguous()  # (K,N) -> (N,K)
        B_e_fp8, B_e_scales = _fp8_blockwise_weight_quant(B_e_T, block_size)

        grad_A[start:end] = fp8_blockwise_gemm(
            grad_e_fp8, B_e_fp8, grad_e_scales, B_e_scales, block_size, out_dtype
        )

        # --- grad_B_e = grad_output_e^T @ A_e ---
        # LHS: (N, M_e_padded), RHS: (M_e_padded, K) -> (N, K)
        # M_e may not be divisible by block_size, so pad.
        pad_m = (block_size - M_e % block_size) % block_size

        grad_e_hp = grad_output[start:end]
        A_e_hp = A_hp[start:end]

        if pad_m > 0:
            grad_e_hp = F.pad(grad_e_hp, (0, 0, 0, pad_m))
            A_e_hp = F.pad(A_e_hp, (0, 0, 0, pad_m))

        # LHS: grad_output_e^T (N, M_padded) — act quant with (1, bs)
        grad_e_t = grad_e_hp.t().contiguous()
        grad_t_fp8, grad_t_scales = _fp8_blockwise_act_quant(grad_e_t, block_size)

        # RHS: A_e (M_padded, K) — weight quant with (bs, bs)
        A_e_fp8, A_e_scales = _fp8_blockwise_weight_quant(
            A_e_hp.contiguous(), block_size
        )

        grad_B_NK = fp8_blockwise_gemm(
            grad_t_fp8, A_e_fp8, grad_t_scales, A_e_scales, block_size, out_dtype
        )
        grad_B[e] = grad_B_NK.t()  # (N, K) -> (K, N)

        start = end

    return grad_A, grad_B


# ---------------------------------------------------------------------------
# Emulated path: quant -> dequant -> torch._grouped_mm
# ---------------------------------------------------------------------------


def _forward_emulated(A, B_t, offs, block_size, out_dtype):
    """Forward using emulated path (quant/dequant + torch._grouped_mm)."""
    E, K, N = B_t.shape

    # Quantize A, then dequantize
    A_fp8, A_scales = _fp8_blockwise_act_quant(A.contiguous(), block_size)
    A_dequant = _fp8_blockwise_dequant_act(A_fp8, A_scales, block_size, out_dtype)

    # Quantize each expert's weight, then dequantize
    B_t_dequant = torch.empty(E, K, N, dtype=out_dtype, device=B_t.device)
    for e in range(E):
        w = B_t[e]
        w_fp8, w_scales = _fp8_blockwise_weight_quant(w.contiguous(), block_size)
        B_t_dequant[e] = _fp8_blockwise_dequant_weight(
            w_fp8, w_scales, block_size, out_dtype
        )

    # Make B_t column-major per expert for grouped mm
    B_t_col = B_t_dequant.contiguous().transpose(-2, -1).contiguous().transpose(-2, -1)

    return torch._grouped_mm(A_dequant, B_t_col, offs=offs, out_dtype=out_dtype)


def _backward_emulated(grad_output, A, B_t, offs, block_size, out_dtype):
    """Backward using emulated path (quant/dequant + torch._grouped_mm)."""
    E, K, N = B_t.shape
    B_hp = B_t._data if hasattr(B_t, "_data") else B_t
    A_hp = A._data if hasattr(A, "_data") else A

    # --- grad_A = grad_output @ B ---
    grad_fp8, grad_scales = _fp8_blockwise_act_quant(
        grad_output.contiguous(), block_size
    )
    grad_dequant = _fp8_blockwise_dequant_act(
        grad_fp8, grad_scales, block_size, out_dtype
    )

    B_dequant = torch.empty(E, N, K, dtype=out_dtype, device=B_t.device)
    for e in range(E):
        w = B_hp[e].transpose(-2, -1).contiguous()  # (K,N) -> (N,K)
        w_fp8, w_scales = _fp8_blockwise_weight_quant(w, block_size)
        B_dequant[e] = _fp8_blockwise_dequant_weight(
            w_fp8, w_scales, block_size, out_dtype
        )

    B_col = B_dequant.contiguous().transpose(-2, -1).contiguous().transpose(-2, -1)

    grad_A = torch._grouped_mm(grad_dequant, B_col, offs=offs, out_dtype=out_dtype)

    # --- grad_B = grad_output^T @ A ---
    M_total = grad_output.size(0)
    pad_m = (block_size - M_total % block_size) % block_size

    grad_output_padded = grad_output
    if pad_m > 0:
        grad_output_padded = F.pad(grad_output, (0, 0, 0, pad_m))

    grad_output_t = grad_output_padded.t().contiguous()
    grad_t_fp8, grad_t_scales = _fp8_blockwise_act_quant(grad_output_t, block_size)
    grad_t_dequant = _fp8_blockwise_dequant_act(
        grad_t_fp8, grad_t_scales, block_size, out_dtype
    )
    if pad_m > 0:
        grad_t_dequant = grad_t_dequant[:, :M_total]

    A_fp8, A_scales_bw = _fp8_blockwise_act_quant(A_hp.contiguous(), block_size)
    A_dequant_bw = _fp8_blockwise_dequant_act(
        A_fp8, A_scales_bw, block_size, out_dtype
    )

    A_col = A_dequant_bw.t().contiguous().t()

    grad_B = torch._grouped_mm(
        grad_t_dequant, A_col, offs=offs, out_dtype=out_dtype
    )

    return grad_A, grad_B.transpose(-2, -1)
