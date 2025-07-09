# GPU-Kernel-Optimization-with-Triton-on-AMD-with-Fusion

## Overview

This project explores **operator fusion techniques in Triton** for accelerating GEMM (General Matrix Multiply) workloads on my personal **AMD Radeon RX 7900 GRE** GPU. Running Triton on this specific GPU (via ROCm) posed unique compatibility challenges, which I resolved through hours of system-level tuning and compatibility fixes (listed futher down).

The primary goal was to assess how **fusing operations** like bias addition, ReLU, and dropout into a single kernel could improve (or worsen, as we'll see) execution efficiency—especially for **smaller tile sizes** commonly used in real-time applications like CV.

---

## Kernels Benchmarked

Three custom Triton kernels were implemented and benchmarked:

1. **Plain GEMM**  
2. **GEMM + Bias + ReLU**  
3. **GEMM + Bias + ReLU + Dropout**  

Each was tested across three matrix shapes:

- (M, K, N) = (64, 256, 128)  
- (M, K, N) = (128, 256, 256)  
- (M, K, N) = (1024, 1024, 1024)

---

## Motivation

Deep learning workloads often involve **repeated GEMM operations** followed by bias, activation (ReLU), and dropout layers. Standard libraries like `rocBLAS` optimize GEMM itself but don’t fuse these layers, leading to **extra memory traffic** and **latency overhead**.

Triton allows building **custom GPU kernels** with free control over memory layout and fusion—unlocking the potential to reduce memory bandwidth consumption by keeping intermediate results in registers or shared memory.

---

## Setup

**Hardware:** AMD Radeon RX 7900 GRE  
**OS:** Ubuntu 22.04.5 LTS  (Important for compatibility for 7900 GRE)
**Kernel:** 6.8.x  
**Software:** ROCm 6.0.2, Triton, PyTorch, Python 3.10  (Important for compatibility for 7900 GRE)
**Tile Size:** `BLOCK_M = BLOCK_N = BLOCK_K = 64`  
**Metric:** Execution time (ms), averaged over ~100 runs per kernel

---

## Results

### Benchmark: M = 64, K = 256, N = 128
- Plain GEMM: **0.105 ms**
- GEMM + Bias + ReLU: **0.101 ms**
- GEMM + Bias + ReLU + Dropout: **0.106 ms**

### Benchmark: M = 128, K = 256, N = 256
- Plain GEMM: **0.087 ms**
- GEMM + Bias + ReLU: **0.085 ms**
- GEMM + Bias + ReLU + Dropout: **0.089 ms**

### Benchmark: M = 1024, K = 1024, N = 1024
- Plain GEMM: **0.328 ms**
- GEMM + Bias + ReLU: **0.327 ms**
- GEMM + Bias + ReLU + Dropout: **0.343 ms**

![Graph of Execution Times](https://github.com/user-attachments/assets/fb7a7c32-e758-4822-8bc2-1fcd936ccc9f)

---

## Analysis

On average, **fusing Bias + ReLU** reduced execution time by **2.14%**. However, **adding Dropout** increased execution time by **2.61%**.

Interestingly, **fusion gains decreased with larger matrix sizes**:

| Matrix Size        | Bias + ReLU Speedup |
|--------------------|---------------------|
| 64 × 256 × 128     | **3.81%**           |
| 128 × 256 × 256    | **2.30%**           |
| 1024 × 1024 × 1024 | **0.30%**           |

---

## Takeaways & Tradeoffs

- **Operator fusion** helps reduce **global memory traffic**, especially on smaller matrices where the relative cost of memory access is high.
- However, **fusing Dropout** introduced additional register pressure and resource contention, leading to **diminished returns** or even slowdowns on larger workloads.
- This exposes an important tradeoff in GPU systems design:  
  > *Reducing memory bandwidth via fusion vs. increasing register pressure and kernel complexity.*

---

## Conclusion

This experiment demonstrates that **Triton operator fusion can yield tangible performance gains**, especially in inference-heavy pipelines with small batch sizes. While these improvements seem modest at best, they can compount significantly when scaled to the billions of computations we see modern deep learning architectures do.

Future work I'm hoping to do:
- Autotuning tile sizes
- Comparing against `rocBLAS`/`torch.matmul`
- Profiling register usage and shared memory occupancy

---

## References

- [Triton Documentation](https://github.com/openai/triton)  
- [ROCm Official Docs](https://rocmdocs.amd.com)

