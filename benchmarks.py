import torch
import triton
import triton.language as tl
import time

# ------------------------------
# Kernel 1: Plain GEMM
# ------------------------------
@triton.jit
def kernel_gemm(
	A_ptr, B_ptr, C_ptr,
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
	tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)

# ------------------------------
# Kernel 2: GEMM + Bias + ReLU
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
# Kernel 3: GEMM + Bias + ReLU + Dropout
# ------------------------------
@triton.jit
def kernel_gemm_bias_relu_dropout(
	A_ptr, B_ptr, bias_ptr, C_ptr,
	M, N, K,
	stride_am, stride_ak,
	stride_bk, stride_bn,
	stride_cm, stride_cn,
	BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
	DROPOUT_P: tl.constexpr
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
	rng_seed = pid_m * 17 + pid_n * 31
	random_vals = tl.rand(rng_seed, BLOCK_M, BLOCK_N)
	mask = random_vals > DROPOUT_P
	acc = acc * mask / (1.0 - DROPOUT_P)
	tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc)

# ------------------------------
# Benchmark Helper
# ------------------------------
def benchmark_kernel(fn, args, grid, **kwargs):
	for _ in range(3):
    	fn[grid](*args, **kwargs)
	torch.cuda.synchronize()
	start = time.time()
	fn[grid](*args, **kwargs)
	torch.cuda.synchronize()
	return (time.time() - start) * 1e3  # ms

# ------------------------------
# Main
# ------------------------------
def run():
	shapes = [(64, 256, 128), (128, 256, 256), (1024, 1024, 1024)]
	BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 64
	DROPOUT_P = 0.1
	device = 'cuda'

	for (M, K, N) in shapes:
    	print(f"\nBenchmarking GEMM shape M={M}, K={K}, N={N}")
    	A = torch.randn((M, K), device=device, dtype=torch.float32)
    	B = torch.randn((K, N), device=device, dtype=torch.float32)
    	bias = torch.randn((N,), device=device, dtype=torch.float32)
    	C = torch.empty((M, N), device=device, dtype=torch.float32)
    	grid = lambda META: (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))

    	t1 = benchmark_kernel(
        	kernel_gemm, [A, B, C, M, N, K,
                      	A.stride(0), A.stride(1),
                      	B.stride(0), B.stride(1),
                      	C.stride(0), C.stride(1)],
        	grid=grid,
        	BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    	)

    	t2 = benchmark_kernel(
        	kernel_gemm_bias_relu, [A, B, bias, C, M, N, K,
                                 	A.stride(0), A.stride(1),
                                 	B.stride(0), B.stride(1),
                                 	C.stride(0), C.stride(1)],
        	grid=grid,
        	BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K
    	)

    	t3 = benchmark_kernel(
        	kernel_gemm_bias_relu_dropout, [A, B, bias, C, M, N, K,
                                        	A.stride(0), A.stride(1),
                                        	B.stride(0), B.stride(1),
                                        	C.stride(0), C.stride(1)],
        	grid=grid,
        	BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        	DROPOUT_P=DROPOUT_P
    	)

    	print(f"[1] Plain GEMM:                	{t1:.3f} ms")
    	print(f"[2] GEMM + Bias + ReLU:       	{t2:.3f} ms")
    	print(f"[3] GEMM + Bias + ReLU + Dropout: {t3:.3f} ms")

if __name__ == "__main__":
	run()
