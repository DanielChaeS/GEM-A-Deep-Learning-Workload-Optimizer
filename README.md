# Triton-GEMM-Kernel-Improvements-via-Operator-Fusion

Overview (aka Abstract):
This project was a result of my curiosity of fusing operator techniques in Triton for GEMM (General Matrix Multiply) workloads on my personal AMD GPU (Radeon RX 7900 GRE) (getting it to run on that GPU is an entire story in itself which I'll get to eventually).

3 Kernels were benchmarked:
1. Plain, boring GEMM
2. GEMM + Bias + ReLU
3. GEMM + Bias + ReLU + Dropout

I also heard that these fusions bring more benefit the smaller the tile sizes are, leading to a potentially useful application in real-time CV, RVC, etc. (I could also be talking out of my *ss who knows)
So, I benchmarked those 3 kernels across 3 matrix shapes (M, K, N) = (64, 256, 128), (128, 256, 256), (1024, 1024, 1024). 
The output of this project was the execution time in ms and a deeper dive into the improvements.

THE REASON FOR DOING THIS:
Training/Learning workloads often consist of TONS of GEMM operations followed by bias addition, activation, dropout, yadda yadda yadda. While libraries like rocBLAS optimize GEMM for PyTorch, they don't fuse these operators that can cause traffic in memory and other overhead. Triton enables custom GPU kernels where fusion can reduce these overhead.

SETUP:
Hardware: AMD Radeon RX 7900 GRE (the goat of mid-range gaming GPUs back when you could find it for MSRP)
Software/Dependencies: Ubuntu 22.04.5 LTS (Kernel: whatever the latest is (6.8.something)), ROCm 6.0.2, Triton, Python 3.10
Static Tile Sizes: BLOCK_M=64, BLOCK_N=64, BLOCK_K=64
Metric: Execution time in ms, averaged over a bunch of runs (100?) I lost count to be honest but it evened out eventually

RESULTS:
Benchmarking GEMM shape M=64, K=256, N=128
[1] Plain GEMM:                	0.105 ms
[2] GEMM + Bias + ReLU:       	0.101 ms
[3] GEMM + Bias + ReLU + Dropout: 0.106 ms

Benchmarking GEMM shape M=128, K=256, N=256
[1] Plain GEMM:                	0.087 ms
[2] GEMM + Bias + ReLU:       	0.085 ms
[3] GEMM + Bias + ReLU + Dropout: 0.089 ms

Benchmarking GEMM shape M=1024, K=1024, N=1024
[1] Plain GEMM:                	0.328 ms
[2] GEMM + Bias + ReLU:       	0.327 ms
[3] GEMM + Bias + ReLU + Dropout: 0.343 ms
![image](https://github.com/user-attachments/assets/62843537-dbc8-4e52-b4af-84f60ee93cd4)

On average, there was a 2.14% DECREASE (YAY!!!) in kernel execution time when fusing Bias + ReLU to the original GEMM. However, adding Dropout to the mix had the opposite effect and instead increased the time by 2.61% on average.
Also, the improvements from Bias + ReLU Fusion gradually decreased as the matrix sizes increased. Going top down, improvements went from 3.81% to 2.30% to 0.30%.

Conclusion:
Operator fusion in Triton can incrementally acelerate deep-learning inference pipelines, particularly for smaller GEMMs. Fusing Dropout is undesirable for reducing execution time but may reduce other overhead; this tradeoff needs to be looked more in depth. It's important to note that while these improvements are small in number, when scaled to millions and billions of calculations, every tiny improvement matters.

References:
Triton Documentation: https://github.com/openai/triton
ROCm: https://rocmdocs.amd.com



