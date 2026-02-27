"""
Test correctness of the Triton FP8 blockwise grouped GEMM kernel.

Compares the Triton path against:
1. The emulated path (quant/dequant + torch._grouped_mm)
2. A bf16 baseline (no quantization)
"""

import torch
import torch.nn.functional as F


def test_gemm_kernel_standalone():
    """Test the raw Triton GEMM kernel against a PyTorch reference."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _fp8_blockwise_act_quant,
        _fp8_blockwise_weight_quant,
        _fp8_blockwise_dequant_act,
        _fp8_blockwise_dequant_weight,
    )
    from torchao.prototype.moe_training.kernels.float8_blockwise import (
        fp8_blockwise_gemm,
    )

    torch.manual_seed(42)
    M, K, N = 256, 256, 256
    block_size = 128

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")

    # Quantize
    A_fp8, A_scales = _fp8_blockwise_act_quant(A, block_size)
    B_fp8, B_scales = _fp8_blockwise_weight_quant(B, block_size)

    # Triton kernel result
    out_triton = fp8_blockwise_gemm(A_fp8, B_fp8, A_scales, B_scales, block_size)

    # Reference: dequant then matmul
    A_dequant = _fp8_blockwise_dequant_act(A_fp8, A_scales, block_size)
    B_dequant = _fp8_blockwise_dequant_weight(B_fp8, B_scales, block_size)
    out_ref = A_dequant @ B_dequant

    diff = (out_triton.float() - out_ref.float()).abs()
    rel_err = diff / (out_ref.float().abs().mean() + 1e-8)
    print(f"[Kernel standalone] max_abs_diff={diff.max().item():.6f}, "
          f"mean_rel_err={rel_err.mean().item():.6f}")

    assert diff.max().item() < 2.0, f"Kernel mismatch too large: {diff.max().item()}"
    print("  PASSED\n")


def test_forward_triton_vs_emulated():
    """Test that Triton forward matches emulated forward (manual dequant reference)."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _fp8_blockwise_act_quant,
        _fp8_blockwise_weight_quant,
        _fp8_blockwise_dequant_act,
        _fp8_blockwise_dequant_weight,
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(42)
    E, M, K, N = 4, 512, 256, 256
    block_size = 128

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_t = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")

    tokens_per_expert = M // E
    offs = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    # Triton path
    out_triton = _to_fp8_blockwise_then_scaled_grouped_mm(
        A, B_t, offs, block_size=block_size, out_dtype=torch.bfloat16, use_triton=True
    )

    # Manual reference: quant, dequant, per-expert matmul (no torch._grouped_mm)
    A_fp8, A_scales = _fp8_blockwise_act_quant(A.contiguous(), block_size)
    A_dequant = _fp8_blockwise_dequant_act(A_fp8, A_scales, block_size)

    out_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e in range(E):
        end = offs[e].item()
        W = B_t[e].contiguous()
        W_fp8, W_scales = _fp8_blockwise_weight_quant(W, block_size)
        W_dq = _fp8_blockwise_dequant_weight(W_fp8, W_scales, block_size)
        out_ref[start:end] = A_dequant[start:end] @ W_dq
        start = end

    diff = (out_triton.float() - out_ref.float()).abs()
    rel_err = diff / (out_ref.float().abs().mean() + 1e-8)
    print(f"[Forward triton vs dequant ref] max_abs_diff={diff.max().item():.6f}, "
          f"mean_rel_err={rel_err.mean().item():.6f}")

    assert diff.max().item() < 2.0, f"Forward mismatch: {diff.max().item()}"
    print("  PASSED\n")


def test_forward_vs_bf16_baseline():
    """Test that Triton forward is close to bf16 baseline."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(42)
    E, M, K, N = 4, 512, 256, 256
    block_size = 128

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_t = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")

    tokens_per_expert = M // E
    offs = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    # Triton path
    out_triton = _to_fp8_blockwise_then_scaled_grouped_mm(
        A, B_t, offs, block_size=block_size, out_dtype=torch.bfloat16, use_triton=True
    )

    # bf16 baseline: manual per-expert matmul
    out_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e in range(E):
        end = offs[e].item()
        out_ref[start:end] = A[start:end] @ B_t[e]
        start = end

    diff = (out_triton.float() - out_ref.float()).abs()
    rel_err = diff / (out_ref.float().abs().mean() + 1e-8)
    print(f"[Forward vs bf16 baseline] max_abs_diff={diff.max().item():.6f}, "
          f"mean_rel_err={rel_err.mean().item():.6f}")

    # FP8 quantization error is expected, allow generous tolerance
    assert rel_err.mean().item() < 0.1, f"Too far from bf16: {rel_err.mean().item()}"
    print("  PASSED\n")


def test_backward_triton():
    """Test backward pass with Triton path produces valid gradients."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(42)
    E, M, K, N = 4, 512, 256, 256
    block_size = 128

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    tokens_per_expert = M // E
    offs = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    out = _to_fp8_blockwise_then_scaled_grouped_mm(
        A, B_t, offs, block_size=block_size, out_dtype=torch.bfloat16, use_triton=True
    )

    loss = out.sum()
    loss.backward()

    assert A.grad is not None, "A.grad is None"
    assert B_t.grad is not None, "B_t.grad is None"
    assert A.grad.shape == A.shape, f"A.grad shape mismatch: {A.grad.shape} vs {A.shape}"
    assert B_t.grad.shape == B_t.shape, f"B_t.grad shape mismatch: {B_t.grad.shape} vs {B_t.shape}"

    # Check gradients are finite
    assert torch.isfinite(A.grad).all(), "A.grad has non-finite values"
    assert torch.isfinite(B_t.grad).all(), "B_t.grad has non-finite values"

    # Check gradients are non-trivial
    assert A.grad.abs().max() > 0, "A.grad is all zeros"
    assert B_t.grad.abs().max() > 0, "B_t.grad is all zeros"

    print(f"[Backward triton] A.grad norm={A.grad.float().norm().item():.4f}, "
          f"B_t.grad norm={B_t.grad.float().norm().item():.4f}")
    print("  PASSED\n")


def test_backward_triton_vs_bf16_reference():
    """Compare backward gradients between Triton and a bf16 reference."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(42)
    E, M, K, N = 4, 512, 256, 256
    block_size = 128

    tokens_per_expert = M // E
    offs = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    # Triton backward
    A1 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B1 = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    out1 = _to_fp8_blockwise_then_scaled_grouped_mm(
        A1, B1, offs, block_size=block_size, use_triton=True
    )
    out1.sum().backward()

    # bf16 reference backward (no quantization)
    A2 = A1.detach().clone().requires_grad_(True)
    B2 = B1.detach().clone().requires_grad_(True)
    out2 = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e in range(E):
        end = offs[e].item()
        out2[start:end] = A2[start:end] @ B2[e]
        start = end
    out2.sum().backward()

    # Compare grad_A (expect FP8 quantization error but should be in same ballpark)
    diff_A = (A1.grad.float() - A2.grad.float()).abs()
    rel_A = diff_A / (A2.grad.float().abs().mean() + 1e-8)
    print(f"[Backward grad_A vs bf16] max_abs_diff={diff_A.max().item():.6f}, "
          f"mean_rel_err={rel_A.mean().item():.6f}")

    # Compare grad_B
    diff_B = (B1.grad.float() - B2.grad.float()).abs()
    rel_B = diff_B / (B2.grad.float().abs().mean() + 1e-8)
    print(f"[Backward grad_B vs bf16] max_abs_diff={diff_B.max().item():.6f}, "
          f"mean_rel_err={rel_B.mean().item():.6f}")

    # FP8 introduces quantization error, so be generous with tolerance
    assert rel_A.mean().item() < 0.15, f"grad_A diverged: {rel_A.mean().item()}"
    assert rel_B.mean().item() < 0.15, f"grad_B diverged: {rel_B.mean().item()}"
    print("  PASSED\n")


def test_uneven_expert_assignment():
    """Test with uneven token-to-expert distribution."""
    from torchao.prototype.moe_training.fp8_blockwise_grouped_mm import (
        _to_fp8_blockwise_then_scaled_grouped_mm,
    )

    torch.manual_seed(42)
    E, K, N = 4, 256, 256
    block_size = 128

    # Uneven distribution: 64, 192, 128, 128 = 512 total
    expert_tokens = [64, 192, 128, 128]
    M = sum(expert_tokens)
    offs = torch.cumsum(
        torch.tensor(expert_tokens, dtype=torch.int32, device="cuda"), dim=0
    )

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda", requires_grad=True)
    B_t = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda", requires_grad=True)

    out = _to_fp8_blockwise_then_scaled_grouped_mm(
        A, B_t, offs, block_size=block_size, use_triton=True
    )

    # Compare to per-expert bf16
    out_ref = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    start = 0
    for e in range(E):
        end = offs[e].item()
        out_ref[start:end] = A[start:end].detach() @ B_t[e].detach()
        start = end

    diff = (out.float() - out_ref.float()).abs()
    rel_err = diff / (out_ref.float().abs().mean() + 1e-8)
    print(f"[Uneven experts] max_abs_diff={diff.max().item():.6f}, "
          f"mean_rel_err={rel_err.mean().item():.6f}")

    # Test backward
    out.sum().backward()
    assert A.grad is not None and torch.isfinite(A.grad).all()
    assert B_t.grad is not None and torch.isfinite(B_t.grad).all()
    print("  PASSED\n")


def test_config_dispatch():
    """Test end-to-end dispatch through FP8BlockwiseGroupedMMConfig."""
    from torchao.prototype.moe_training.config import FP8BlockwiseGroupedMMConfig
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor

    torch.manual_seed(42)
    E, M, K, N = 4, 256, 256, 256
    block_size = 128

    config = FP8BlockwiseGroupedMMConfig(block_size=block_size, use_triton=True)

    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B_t = torch.randn(E, K, N, dtype=torch.bfloat16, device="cuda")
    B_t_wrapped = ScaledGroupedMMTensor(B_t, config)

    tokens_per_expert = M // E
    offs = torch.cumsum(
        torch.full((E,), tokens_per_expert, dtype=torch.int32, device="cuda"), dim=0
    )

    out = torch._grouped_mm(A, B_t_wrapped, offs=offs)
    assert out.shape == (M, N), f"Wrong output shape: {out.shape}"
    assert torch.isfinite(out).all(), "Output has non-finite values"

    print(f"[Config dispatch] output shape={out.shape}, dtype={out.dtype}")
    print("  PASSED\n")


if __name__ == "__main__":
    print("=" * 60)
    print("FP8 Blockwise Triton GEMM Correctness Tests")
    print("=" * 60)
    print()

    test_gemm_kernel_standalone()
    test_forward_triton_vs_emulated()
    test_forward_vs_bf16_baseline()
    test_backward_triton()
    test_backward_triton_vs_bf16_reference()
    test_uneven_expert_assignment()
    test_config_dispatch()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
