# GPU-Kernel-Optimization-with-Triton-on-AMD-with-Fusion

## Overview

This project explores **operator fusion techniques in Triton** for accelerating GEMM (General Matrix Multiply) workloads on my personal **AMD Radeon RX 7900 GRE** GPU. Running Triton on this specific GPU (via ROCm) posed unique compatibility challenges, which I resolved through hours of system-level tuning and compatibility fixes (listed futher down).

The primary goal was to assess how **fusing operations** like bias addition, ReLU, and dropout into a single kernel could improve (or worsen, as we'll see) execution efficiency—especially for **smaller tile sizes** commonly used in real-time applications like CV.

---

## Kernels Benchmarked

Two custom Triton kernels were implemented and benchmarked:

1. **GEMM + Bias + ReLU**
2. **FUSED GEMM + Bias + ReLU**

Each was tested across three matrix shapes:

- (M, K, N) = (64, 256, 128)  
- (M, K, N) = (128, 256, 256)  
- (M, K, N) = (1024, 1024, 1024)

---

## Motivation

Deep learning workloads often involve **repeated GEMM operations** followed by bias and activation (ReLU). Standard libraries like `rocBLAS` optimize GEMM itself but don’t fuse these layers, leading to **extra memory traffic**. Think: we have to refer constantly to a matrix C after performing matmul on A and B. This matrix C is normally stored in global memory which can take hundreds of cycles to access!!! Now, what if we could keep the value of C in registers/shared memory where the program can access it in 20 cycles or less? (BTW there isn't enough space on registers to only use registers boohoo)

Triton allows building **custom GPU kernels** with free control over memory layout and fusion—unlocking the potential to reduce memory bandwidth consumption by keeping intermediate results in registers or shared memory.

---

## Setup

**Hardware:** AMD Radeon RX 7900 GRE  
**OS:** Ubuntu 22.04.5 LTS  (Important for compatibility for 7900 GRE)
**Kernel:** 6.8.x  
**Software:** ROCm 6.0.2, Triton, PyTorch, Python 3.10  (Important for compatibility for 7900 GRE) (Also, turn off Secure Boot!!!)
**Tile Size:** `BLOCK_M = BLOCK_N = BLOCK_K = 64`  
**Metric:** Execution time (ms), averaged over ~100 runs per kernel

---

## Results

### Benchmark: M = 64, K = 256, N = 128
- GEMM + Bias + ReLU: **0.165 ms**
- FUSED GEMM + Bias + ReLU: **0.101 ms**

### Benchmark: M = 128, K = 256, N = 256
- GEMM + Bias + ReLU: **0.197 ms**
- FUSED GEMM + Bias + ReLU: **0.131 ms**

### Benchmark: M = 1024, K = 1024, N = 1024
- GEMM + Bias + ReLU: **0.460 ms**
- FUSED GEMM + Bias + ReLU: **0.327 ms**


<img width="1697" height="1101" alt="image" src="https://github.com/user-attachments/assets/8d20a290-5821-4c89-bbd6-84dde51419fa" />


---

## Analysis

On average, **fusing Bias + ReLU** reduced execution time by **33.7%**:

| Matrix Size        | Bias + ReLU Speedup |
|--------------------|---------------------|
| 64 × 256 × 128     | **38.8%**           |
| 128 × 256 × 256    | **33.5%**           |
| 1024 × 1024 × 1024 | **28.9%**           |

---

## Takeaways & Tradeoffs

- **Operator fusion** helps reduce **global memory traffic**.
- **GPU Memory Hierarchy** is very similar to **CPU Cache Hierarchy**. Difference is that GPU registers are bigger
- Fusion **has promise**, but might see decline in efficiency due to overhead on bigger workloads (think in the millions). 

---

## Conclusion

This experiment demonstrates that **Triton operator fusion can yield tangible performance gains**, especially in inference-heavy pipelines with small batch sizes.

Future work I'm hoping to do:
- Testing larger tile sizes
- Profiling register/shared memory usage and finding the break-even point

---

## References

- [Triton Documentation](https://github.com/openai/triton)  
- [ROCm Official Docs](https://rocmdocs.amd.com)

