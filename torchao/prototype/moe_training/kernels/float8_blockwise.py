# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD 3-Clause license found in the
# LICENSE file in the root directory of this source tree.

"""
Triton FP8 blockwise GEMM kernel for MoE training.

Fuses blockwise dequantization into the GEMM accumulation loop, so the
full dequantized tensors are never materialized in global memory.

Scale granularity (following DeepSeek-V3):
  - LHS (activations): (1, block_size) — one scale per row per K-block
  - RHS (weights):     (block_size, block_size) — one scale per tile

The kernel accumulates in float32 and casts the result to the output dtype.

The inner loop processes one K-block per iteration:
    for each K-block kb:
        acc += dot(A_fp8[..., kb], B_fp8[kb, ...]) * a_scale[:, kb] * b_scale[kb, :]

This is mathematically equivalent to dequantizing then matmul, but avoids
the O(M*K + K*N) memory traffic of materializing the dequantized tensors.
"""

import math

import torch

from torchao.utils import torch_version_at_least

try:
    from torch.utils._triton import has_triton
except ImportError:

    def has_triton():
        return False


if torch_version_at_least("2.7.0") and has_triton():
    import triton
    import triton.language as tl

    _gemm_configs = [
        triton.Config(
            {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn},
            num_warps=warps,
            num_stages=stages,
        )
        for bm in [32, 64, 128]
        for bn in [64, 128]
        for warps in [4, 8]
        for stages in [2, 4]
    ]

    @triton.autotune(configs=_gemm_configs, key=["N", "K", "M_BUCKET", "BLOCK_SIZE_K"])
    @triton.jit
    def _fp8_blockwise_gemm_kernel(
        # A: (M, K) in FP8
        a_ptr,
        stride_am,
        stride_ak,
        # B: (K, N) in FP8
        b_ptr,
        stride_bk,
        stride_bn,
        # C: (M, N) output
        c_ptr,
        stride_cm,
        stride_cn,
        # A scales: (M, K // BLOCK_SIZE_K), dequantization multipliers
        a_s_ptr,
        stride_as_m,
        stride_as_kb,
        # B scales: (K // BLOCK_SIZE_K, N // BLOCK_SIZE_K), dequantization multipliers
        b_s_ptr,
        stride_bs_kb,
        stride_bs_nb,
        M,
        N: tl.constexpr,
        K: tl.constexpr,
        M_BUCKET: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)

        # Pointers to the first (BLOCK_SIZE_M, BLOCK_SIZE_K) tile of A and
        # the first (BLOCK_SIZE_K, BLOCK_SIZE_N) tile of B.
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # Scale base pointers. For A, each row m has one scale per K-block.
        # For B, each (K-block, N-block) tile has one scale. The N-block index
        # for each element in this tile is offs_n // BLOCK_SIZE_K.
        a_s_base = a_s_ptr + offs_m * stride_as_m
        b_s_base = b_s_ptr + (offs_n // BLOCK_SIZE_K) * stride_bs_nb

        k_num_blocks = tl.cdiv(K, BLOCK_SIZE_K)
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

        for kb in range(k_num_blocks):
            k_remaining = K - kb * BLOCK_SIZE_K
            a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
            b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)

            # Load dequant scales: a_s is (BLOCK_SIZE_M,), b_s is (BLOCK_SIZE_N,).
            # When BLOCK_SIZE_N == BLOCK_SIZE_K, all b_s elements within a tile
            # map to the same weight scale block, so b_s is effectively a scalar
            # broadcast. The kernel handles arbitrary BLOCK_SIZE_N correctly via
            # the per-element offs_n // BLOCK_SIZE_K indexing.
            a_s = tl.load(
                a_s_base + kb * stride_as_kb, mask=offs_m < M, other=1.0
            )
            b_s = tl.load(
                b_s_base + kb * stride_bs_kb, mask=offs_n < N, other=1.0
            )

            # FP8 dot product (uses tensor/matrix cores where available),
            # then rescale by the blockwise dequantization factors.
            accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]

            a_ptrs += BLOCK_SIZE_K * stride_ak
            b_ptrs += BLOCK_SIZE_K * stride_bk

        c = accumulator.to(c_ptr.dtype.element_ty)
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

    def fp8_blockwise_gemm(
        a: torch.Tensor,
        b: torch.Tensor,
        a_scales: torch.Tensor,
        b_scales: torch.Tensor,
        block_size: int = 128,
        out_dtype: torch.dtype = torch.bfloat16,
    ) -> torch.Tensor:
        """
        FP8 GEMM with blockwise dequantization fused into accumulation.

        Computes C = A @ B where A and B are FP8 tensors with per-block scales.
        Scales are applied inside the K-dimension loop so the full dequantized
        matrices are never materialized.

        Args:
            a: (M, K) FP8 tensor.
            b: (K, N) FP8 tensor.
            a_scales: (M, K // block_size) float32 dequant scales (= amax / fp8_max).
            b_scales: (K // block_size, N // block_size) float32 dequant scales.
            block_size: Quantization tile size along K (default 128).
            out_dtype: Output dtype (default bfloat16).

        Returns:
            (M, N) tensor in out_dtype.
        """
        M, K = a.shape
        K2, N = b.shape
        assert K == K2, f"K mismatch: a has {K}, b has {K2}"
        assert K % block_size == 0, f"K={K} not divisible by block_size={block_size}"

        M_BUCKET = math.ceil(math.log2(max(M, 1)))
        c = torch.empty(M, N, dtype=out_dtype, device=a.device)

        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

        _fp8_blockwise_gemm_kernel[grid](
            a,
            a.stride(0),
            a.stride(1),
            b,
            b.stride(0),
            b.stride(1),
            c,
            c.stride(0),
            c.stride(1),
            a_scales,
            a_scales.stride(0),
            a_scales.stride(1),
            b_scales,
            b_scales.stride(0),
            b_scales.stride(1),
            M,
            N,
            K,
            M_BUCKET,
            BLOCK_SIZE_K=block_size,
        )
        return c

else:

    def fp8_blockwise_gemm(*args, **kwargs):
        raise NotImplementedError(
            "fp8_blockwise_gemm requires torch 2.7.0+ and triton"
        )
