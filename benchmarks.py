import torch
import triton
import triton.language as tl
import time

# ------------------------------
# Fused Kernel: GEMM + Bias + ReLU
# ------------------------------
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
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        acc += tl.dot(a, b)

    bias = tl.load(bias_ptr + offs_n)
    acc += bias[None, :]
    acc = tl.maximum(acc, 0)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)

# ------------------------------
# Benchmarking Utility
# ------------------------------
def benchmark(fn, args, warmup=3, repeat=10):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeat):
        fn(*args)
    torch.cuda.synchronize()
    return (time.time() - start) * 1e3 / repeat  # return ms per run

# ------------------------------
# Main Benchmarking Script
# ------------------------------
def run():
    shapes = [(64, 256, 128), (128, 256, 256), (1024, 1024, 1024)]
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
    device = 'cuda'

    for (M, K, N) in shapes:
        print(f"\nBenchmarking M={M}, K={K}, N={N}")

        A = torch.randn((M, K), device=device, dtype=torch.float32)
        B = torch.randn((K, N), device=device, dtype=torch.float32)
        bias = torch.randn((N,), device=device, dtype=torch.float32)
        C = torch.empty((M, N), device=device, dtype=torch.float32)

        # Triton Fused GEMM + Bias + ReLU
        grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        fused_time = benchmark(
            lambda: kernel_gemm_bias_relu[grid](
                A, B, bias, C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
            )
        )

        # Unfused: GEMM (Triton) + Bias (PyTorch) + ReLU (PyTorch)
        def run_unfused():
            kernel_gemm_bias_relu[grid](  # reuse fused kernel with bias=0, relu=False
                A, B, torch.zeros_like(bias), C, M, N, K,
                A.stride(0), A.stride(1),
                B.stride(0), B.stride(1),
                C.stride(0), C.stride(1),
                BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
            )
            C += bias
            torch.relu_(C)

        unfused_time = benchmark(run_unfused)

        print(f"Unfused GEMM + Bias + ReLU: {unfused_time:.3f} ms")
        print(f"Fused GEMM + Bias + ReLU:   {fused_time:.3f} ms")

if __name__ == "__main__":
    run()
