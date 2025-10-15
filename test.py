"""
Test suite for GEMM + Bias + ReLU kernel
Run with: pytest test_gemm.py -v
"""

import pytest
import torch
import triton
from gemm_optimizer import kernel_gemm_bias_relu

# ============================================================================
# Fixtures and Utilities
# ============================================================================

@pytest.fixture
def device():
    """Ensure CUDA is available"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return 'cuda'

def run_triton_kernel(A, B, bias):
    """Helper to run Triton kernel"""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, "Dimension mismatch"
    
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    
    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_M']),
            triton.cdiv(N, META['BLOCK_N'])
        )
    
    kernel_gemm_bias_relu[grid](
        A, B, bias, C, M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    
    return C

def run_pytorch_reference(A, B, bias):
    """Reference implementation using PyTorch"""
    C = torch.mm(A, B)
    C = C + bias
    C = torch.relu(C)
    return C

# ============================================================================
# Correctness Tests
# ============================================================================

class TestCorrectness:
    """Test numerical correctness against PyTorch"""
    
    @pytest.mark.parametrize("M,K,N", [
        (1, 1, 1),              # Minimum size
        (7, 11, 13),            # Small primes (non-block-aligned)
        (64, 64, 64),           # Exact single block
        (65, 65, 65),           # Just over one block
        (128, 128, 128),        # Multiple blocks
        (100, 200, 150),        # Non-square, non-aligned
        (256, 512, 1024),       # Large non-square
        (1024, 1024, 1024),     # Large square
    ])
    def test_various_shapes(self, M, K, N, device):
        """Test correctness across different matrix dimensions"""
        torch.manual_seed(42)
        
        A = torch.randn((M, K), device=device, dtype=torch.float32)
        B = torch.randn((K, N), device=device, dtype=torch.float32)
        bias = torch.randn((N,), device=device, dtype=torch.float32)
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3), \
            f"Mismatch for shape ({M}, {K}, {N}). Max diff: {(C_triton - C_torch).abs().max():.6f}"
    
    def test_zero_bias(self, device):
        """Test with zero bias"""
        M, K, N = 64, 64, 64
        A = torch.randn((M, K), device=device)
        B = torch.randn((K, N), device=device)
        bias = torch.zeros((N,), device=device)
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)
    
    def test_negative_inputs(self, device):
        """Test ReLU activation with negative values"""
        M, K, N = 32, 32, 32
        A = torch.randn((M, K), device=device) - 1.0  # Shift negative
        B = torch.randn((K, N), device=device) - 1.0
        bias = torch.randn((N,), device=device) - 1.0
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        # Verify ReLU is working (should have zeros)
        assert (C_triton == 0).any(), "ReLU should produce some zeros"
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)
    
    def test_all_positive(self, device):
        """Test with all positive inputs"""
        M, K, N = 32, 32, 32
        A = torch.abs(torch.randn((M, K), device=device)) + 0.1
        B = torch.abs(torch.randn((K, N), device=device)) + 0.1
        bias = torch.abs(torch.randn((N,), device=device)) + 0.1
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        # No zeros expected
        assert (C_triton > 0).all(), "Should have no zeros with positive inputs"
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)

# ============================================================================
# Edge Case Tests
# ============================================================================

class TestEdgeCases:
    """Test boundary conditions and edge cases"""
    
    def test_single_element(self, device):
        """Test 1x1x1 matrix multiplication"""
        A = torch.tensor([[2.0]], device=device)
        B = torch.tensor([[3.0]], device=device)
        bias = torch.tensor([1.0], device=device)
        
        C_triton = run_triton_kernel(A, B, bias)
        expected = torch.tensor([[7.0]], device=device)  # 2*3 + 1 = 7
        
        assert torch.allclose(C_triton, expected, atol=1e-5)
    
    def test_one_dimension_one(self, device):
        """Test when one dimension is 1"""
        # MxK=1
        A = torch.randn((1, 100), device=device)
        B = torch.randn((100, 50), device=device)
        bias = torch.randn((50,), device=device)
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)
    
    def test_extreme_aspect_ratio(self, device):
        """Test very wide/tall matrices"""
        # Very wide
        A = torch.randn((16, 32), device=device)
        B = torch.randn((32, 512), device=device)
        bias = torch.randn((512,), device=device)
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        assert torch.allclose(C_triton, C_torch, atol=1e-3, rtol=1e-3)

# ============================================================================
# Numerical Stability Tests
# ============================================================================

class TestNumericalStability:
    """Test with extreme values"""
    
    def test_large_values(self, device):
        """Test with large magnitude values"""
        M, K, N = 64, 64, 64
        A = torch.randn((M, K), device=device) * 100
        B = torch.randn((K, N), device=device) * 100
        bias = torch.randn((N,), device=device) * 100
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        # Check no NaN/Inf
        assert not torch.isnan(C_triton).any(), "NaN detected"
        assert not torch.isinf(C_triton).any(), "Inf detected"
        
        # Relative tolerance needed for large values
        assert torch.allclose(C_triton, C_torch, atol=1e-1, rtol=1e-2)
    
    def test_small_values(self, device):
        """Test with small magnitude values"""
        M, K, N = 64, 64, 64
        A = torch.randn((M, K), device=device) * 0.001
        B = torch.randn((K, N), device=device) * 0.001
        bias = torch.randn((N,), device=device) * 0.001
        
        C_triton = run_triton_kernel(A, B, bias)
        C_torch = run_pytorch_reference(A, B, bias)
        
        assert torch.allclose(C_triton, C_torch, atol=1e-5, rtol=1e-3)

# ============================================================================
# Performance Sanity Tests
# ============================================================================

class TestPerformance:
    """Basic performance checks (not rigorous benchmarks)"""
    
    def test_faster_than_pytorch(self, device):
        """Sanity check that fusion provides speedup"""
        M, K, N = 1024, 1024, 1024
        A = torch.randn((M, K), device=device)
        B = torch.randn((K, N), device=device)
        bias = torch.randn((N,), device=device)
        
        # Warmup
        for _ in range(5):
            _ = run_triton_kernel(A, B, bias)
            _ = run_pytorch_reference(A, B, bias)
        
        torch.cuda.synchronize()
        
        # Quick timing (not production benchmark)
        import time
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(10):
            _ = run_triton_kernel(A, B, bias)
        end.record()
        torch.cuda.synchronize()
        triton_time = start.elapsed_time(end) / 10
        
        start.record()
        for _ in range(10):
            _ = run_pytorch_reference(A, B, bias)
        end.record()
        torch.cuda.synchronize()
        pytorch_time = start.elapsed_time(end) / 10
        
        speedup = pytorch_time / triton_time
        
        # Should be at least as fast (allow 5% slower for autotuning variance)
        assert speedup > 0.95, f"Too slow: {speedup:.2f}x (expected >0.95x)"
        print(f"\n  Speedup: {speedup:.2f}x ({triton_time:.3f}ms vs {pytorch_time:.3f}ms)")

# ============================================================================
# Memory Tests
# ============================================================================

class TestMemory:
    """Test memory behavior"""
    
    def test_no_memory_leak(self, device):
        """Run many iterations without OOM"""
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        for _ in range(100):
            A = torch.randn((128, 128), device=device)
            B = torch.randn((128, 128), device=device)
            bias = torch.randn((128,), device=device)
            C = run_triton_kernel(A, B, bias)
            del A, B, bias, C
        
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()
        
        # Memory should return to baseline (within small margin)
        memory_leak = final_memory - initial_memory
        assert memory_leak < 1024 * 1024, f"Memory leak detected: {memory_leak / 1024:.1f} KB"

# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    # Can run directly: python test_gemm.py
    pytest.main([__file__, "-v", "--tb=short"])
