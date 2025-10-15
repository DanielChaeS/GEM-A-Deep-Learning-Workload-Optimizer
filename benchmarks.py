import torch
import triton
import triton.language as tl

# ------------------------------
# Fused Kernel: GEMM + Bias + ReLU with Autotuning
# ------------------------------
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_gemm_bias_relu(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Accumulator in FP32 for numerical stability
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K-dimension loop with proper indexing and masking
    for k in range(0, K, BLOCK_K):
        # Bounds checking masks
        mask_k = (k + offs_k) < K
        mask_a = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_b = (mask_k[:, None]) & (offs_n[None, :] < N)
        
        # Load tiles with masking
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak,
                    mask=mask_a, other=0.0)
        b = tl.load(B_ptr + (k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_b, other=0.0)
        
        # Matrix multiplication accumulation
        acc += tl.dot(a, b)
    
    # Load bias with masking
    mask_n = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc += bias[None, :]
    
    # Fused ReLU activation
    acc = tl.maximum(acc, 0.0)
    
    # Store result with masking
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, 
             acc, mask=mask_c)

# ------------------------------
# Benchmarking Utility with CUDA Events
# ------------------------------
def benchmark(fn, warmup=10, repeat=100):
    """Accurate GPU benchmarking using CUDA events"""
    for _ in range(warmup):
        fn()
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(repeat):
        fn()
    end_event.record()
    
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    
    return elapsed_time / repeat  # ms per run

# ------------------------------
# Main Benchmarking Script
# ------------------------------
def run():
    shapes = [
        (128, 128, 128),
        (512, 512, 512),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
        (256, 512, 1024),  # Non-square
        (1024, 256, 2048),  # Non-square
    ]
    
    device = 'cuda'
    dtype = torch.float32
    
    print("=" * 80)
    print("GEMM + Bias + ReLU Optimization Benchmark")
    print("=" * 80)
    
    for (M, K, N) in shapes:
        print(f"\n{'='*80}")
        print(f"Shape: M={M}, K={K}, N={N}")
        print(f"{'='*80}")
        
        # Initialize tensors
        A = torch.randn((M, K), device=device, dtype=dtype)
        B = torch.randn((K, N), device=device, dtype=dtype)
        bias = torch.randn((N,), device=device, dtype=dtype)
        C_triton = torch.empty((M, N), device=device, dtype=dtype)
        
        # Grid configuration
        def grid(META):
            return (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
        
        # Triton Fused Kernel
        def run_triton():
            kernel_gemm_bias_relu[grid](
                A, B, bias, C_triton, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C_triton.stride(0), C_triton.stride(1),
            )
        
        # PyTorch Baseline (unfused)
        def run_pytorch():
            C_torch = torch.mm(A, B)
            C_torch += bias
            torch.relu_(C_torch)
            return C_torch
        
        # Warmup and correctness check
        run_triton()
        C_torch = run_pytorch()
        
        # Validate correctness
        if torch.allclose(C_triton, C_torch, atol=1e-2, rtol=1e-2):
            print("✓ Correctness check passed")
        else:
            max_diff = (C_triton - C_torch).abs().max().item()
            print(f"✗ Correctness check FAILED! Max difference: {max_diff}")
            continue
        
        # Benchmark
        triton_time = benchmark(run_triton, warmup=10, repeat=100)
        pytorch_time = benchmark(run_pytorch, warmup=10, repeat=100)
        
        # Calculate metrics
        flops = 2 * M * N * K + M * N  # GEMM FLOPs + bias add
        triton_tflops = flops / (triton_time * 1e-3) / 1e12
        pytorch_tflops = flops / (pytorch_time * 1e-3) / 1e12
        speedup = pytorch_time / triton_time
        
        # Print results
        print(f"\nPyTorch (unfused):      {pytorch_time:.3f} ms  ({pytorch_tflops:.2f} TFLOPS)")
        print(f"Triton (fused):         {triton_time:.3f} ms  ({triton_tflops:.2f} TFLOPS)")
        print(f"Speedup:                {speedup:.2f}x")
        print(f"Performance gain:       {(speedup - 1) * 100:.1f}%")

if __name__ == "__main__":
    run()
