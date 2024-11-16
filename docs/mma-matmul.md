---
layout: page
title: "Optimizing a Tensor Core matmul kernel on the Ada Architecture"
permalink: /mma-matmul

---
Using tensor cores is a prerequisite to get anywhere near peak performance matrix multiplication on NVIDIA GPUs. Compared to traditional CUDA programming, there are not many resources demonstrating how to write efficient Tensor Core kernels. There is a canonical open source library (CUTLASS) to learn from but its heavy use of C++ templates means the code can be difficult to parse.

In this post we work through the process of developing an efficient Tensor Core matrix multiplication kernel targeting the Ada architecture. We start with a naive implementation and by incorporating several techniques used in CUTLASS[^3], finish with a kernel that matches cuBLAS performance on one particular problem specification:

| Kernel | Execution Time | TFLOP/s &nbsp; &nbsp; | cuBLAS % &nbsp; &nbsp;  | 4090 peak % &nbsp; &nbsp; &nbsp; &nbsp; &nbsp; |
| ---    | ---     | ---                 | ---                  | --- |
| cublasGemmEx | 890 us | 154.4 | 100% |  93.5%  |
| Kernel 1: Naive MMA | 4.41 ms | 31.2 | 19.5% | 18.9%   |
| Kernel 1b: Naive + 2x tiling| 2.26 ms | 60.8 | 38.1% | 36.8%   |
| Kernel 2: Permuted shared memory layout | 4.3ms | 20 | 19.3% |    |
| Kernel 3: N stage async pipeline | 4.3ms | 20 | 19.3% |    |
| Kernel 4: N stage + 4x tiling | 840 us | 163.6 | 102.3% | 99.0%   |

In the process of doing this we'll learn about the `mma`, `ldmatrix` and `cp.async` PTX instructions, and how to call inline PTX from CUDA code. The code is written as simply as possible: the aim is ease of understanding rather than generality. 

As may be clear already, this post was heavily inspired by Simon Boehm's great post on optimizing a CUDA matmul kernel[^8]. 

## Problem Definition 
We'll focus on one particular set of shapes: M=N=K=4096, and data types: fp16 A/B, fp32 C/D.  The FLOP count of this operation is `2*4096^3 = 137.4 GFLOP`[^2] (convential to count one FMA as 2 FLOP) and the peak fp16/32 throughput of the RTX 4090 is 165.2 TFLOP/s[^6], so the lower bound on kernel execution time is ~830 us.

We can use the RTX 4090s peak throughput to deduce how many cycles one Tensor Core instruction takes to complete (latency). All our kernels will use the PTX `m16n8k16` `mma` instruction, this is the largest Tensor Core matmul supported on Ada so it's reasonable to assume the peak throughput is obtained using this instruction. The m16n8k16 operation is `2*16*8*16=4096` FLOPS, and there are 512 Tensor Cores on the RTX 4090, hence computing one mma on all Tensor Cores gives 2,097,152 FLOPS. Given the peak throughput of 165.2 TFLOPS/S at the boost clock of 2520 MHz, it must take 12.7 ns = 32 cycles for the m16n8k16 `mma` operation to complete. This is roughly consistent with empirical benchmarks[^7].

Our problem shape of M=N=K=4096 requires 256x512x256 = 33,554,432 individual m16n8k16 `mma` instructions, which is 65,536 card-wide waves of `mma`s. Hence in the best case, with no cycles stalled waiting for input, the minimum number of cycles this will take is 65,636 * 32 = 2,097,152, which is 832 us at the boost clock of 2520 MHz. Note this agrees with the number computed using peak throughput by definition as we computed the 32 cycle latency from the throughput. So there's no new information here but thinking about the time in terms of number of instructions and cycles is helpful to understand nsight-compute metrics.

### Benchmarking Setup
As a baseline for peformance we use the cuBLAS `cublasGemmEx` API with `fp16` inputs and`fp32` accumulation. This performs a `M=N=K=4096` matrix multiply in 890 us which is a throughput of 154.4 TFLOPS/s, 93.5% of the RTX 4090's peak.

How to accurately time CUDA kernel execution could fill an entire post but in summary either cuda events or nsight-compute give broadly consistent results if you first lock the gpu and memory clocks. I used nsight-compute as it measures kernel execution more precisely than possible using events [^5]. 

By default nsight-compute will lock to the GPUs base clock, but as I wanted to compare to the RTX 4090s stated peak throughput I aimed to lock at the boost clock of 2520 MHz. This issomewhat difficult in practice as locking at 2520 results in nsight-compute reporting slightly lower clock frequencies. As a result I locked at 2550 and report results based on runs where nsight-compute reports the clock as 2520. The commands to lock clocks and run the profiler are:

```bash
sudo nvidia-smi -pm ENABLED
sudo nvidia-smi --lock-gpu-clocks=2550     # lock slightly higher than boost clock
sudo nvidia-smi --lock-memory-clocks=10501 # max for RTX 4090
ncu -k $my_kernel_name --clock-control none --print-summary per-gpu $my_executable
```

### Aside: Tensor Core Matrix Multiply APIs
There are three separate Tensor Core matmul APIs in CUDA/PTX:
* WMMA: High level API available in both [CUDA](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#warp-matrix-functions) and [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-wmma-instructions)
* MMA: Lower level API just available in [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#matrix-multiply-accumulate-operation-using-mma-instruction)
* WGMMA: New sm_90 API that operates on warp-groups (consecutive groups of 4 warps). Just available in [PTX](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-multiply-accumulate-instructions)

All kernels in this post use the PTX mma API. wgmma is not an option as I am using an Ada architecture GPU. I chose mma over wmma as mma is a lower level API and my aim is to build as deep an understanding of Tensor Core operations as possible. Using mma also reportedly delivers higher performance than wmma though that comparison is old[^1].

## Kernel 1: Naive mma.sync kernel
The first kernel is a naive implementation resulting from reading the `mma.sync` instruction documentation and handling data movement from global memory to registers in the simplest way possible. 

In the Ada architecture there are 4 warp schedulers per SM, each with their own Tensor Core. Hence we want at least 4 warps per thread block (not strictly required as multiple thread blocks can run concurrently on one SM). In this kernel we use a 16x16 thread block, containing 8 warps. Each warp computes one 16x8 output tile and we arrange the warps in a 2 row x 4 column grid, so that each thread block computes a 32x32 output tile.

```c
// arrangement of warps in output tile
// (warp_0 | warp_1 | warp_2 | warp_3)
// (warp_4 | warp_5 | warp_6 | warp_7)
```
There are multiple `mma.sync` instructions for different data types and matrix shapes. As mentioned previously, in this and all subsequent kernels we'll use
- `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` 

which performs (per warp) the matrix multiplication `D = A * B + C` where A is a `16x16` `fp16` matrix, `B` is `16x8` `fp16` matrix and C/D are 16x8 `fp32` matrices. 

As `mma` is a PTX instruction, calling it from CUDA code requires using [inline PTX](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html) which we wrap in a helper function:
{% raw %}
```c++
__device__ void mma_m16n8k16(const unsigned *A, const unsigned *B, float *C, float *D) {
  asm(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
      :
      "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
      "r"(B[0]), "r"(B[1]),
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
      );
}
```
{% endraw %}
The `mma` instruction is warp-wide, each of the 32 threads provides 8 `fp16` elements from A, 4 `fp16` elements from B and 4 `fp32` elements from C, and recieves 4 output `fp32` elements from D. The 8 `fp16` elements of A are packed into 4 32 bit registers, and similarly the 4 elements of B into 2 32 bit registers.

The matrix elements held by each thread in its registers are called a matrix fragment, and the required mapping from thread ID to fragments for A is shown below:
![a-fragment](/assets/images/a-fragment.png)
A is split into 4 8x8 submatrices, and each submatrix is split across the warp in a row major fashion which each thread holding two `fp16` values in one of its 32 bit registers. Mappings for `B, C & D` are defined similarly and can be found in the [PTX docs](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-16816-float).

We will later use the `ldmatrix` instruction to load fragements to registers, but for now we'll do this per thread to demostrate the mapping. The main loop of Kernel 1 contains the code to load matrix fragments to registers and call the `mma` instruction.

```c++
for (int kStart=0; kStart < K; kStart += K_BLOCK) {
  // load from global to shared memory
  for (int m=0; m < 2; ++m) {
    As[m*K_BLOCK + ty][tx] = A[(mBlock + ty + m*K_BLOCK)*K + kStart + tx];
  }
  for (int n=0; n < 2; ++n) {
    Bs[ty][n*K_BLOCK + tx] = B[(kStart + ty) * K + nBlock + n*K_BLOCK + tx];
  }
  __syncthreads();

  // load from shmem to fp16 registers
  aReg[0] = As[mWarp + groupID    ][groupLaneID*2    ];
  aReg[1] = As[mWarp + groupID    ][groupLaneID*2 + 1];
  aReg[2] = As[mWarp + groupID + 8][groupLaneID*2    ];
  aReg[3] = As[mWarp + groupID + 8][groupLaneID*2 + 1];
  aReg[4] = As[mWarp + groupID    ][groupLaneID*2 + 8];
  aReg[5] = As[mWarp + groupID    ][groupLaneID*2 + 9];
  aReg[6] = As[mWarp + groupID + 8][groupLaneID*2 + 8];
  aReg[7] = As[mWarp + groupID + 8][groupLaneID*2 + 9];

  bReg[0] = Bs[groupLaneID*2 + 0][nWarp + groupID];
  bReg[1] = Bs[groupLaneID*2 + 1][nWarp + groupID];
  bReg[2] = Bs[groupLaneID*2 + 8][nWarp + groupID];
  bReg[3] = Bs[groupLaneID*2 + 9][nWarp + groupID];
  // pack fp16 registers to u32 and call mma
  unsigned const *aPtr = reinterpret_cast<unsigned const *>(&aReg);
  unsigned const *bPtr = reinterpret_cast<unsigned const *>(&bReg);
  mma_m16n8k16(aPtr, bPtr, dReg, dReg);
  __syncthreads();
}
```
### Performance
Kernel 1 has an execution time of 4.67 ms, giving a throughput of 29.4 TFLOP/S, 19.1% of cuBLAS and 17.8% of peak RTX 4090 performance. In fact Kernel 1 only achieves 35.6% of the RTX 4090's peak FP32 performance, so a reasonably optimized non Tensor Core kernel would be faster. The reasons for the poor performance are:
1. Each thread loads individual 16b values which is very inefficient. Also the global load pattern is not coalesced. 
2. The loads from shared memory to registers have multiple bank conflicts. 
3. Each element loaded is only used in the input to one `mma` instruction, so the ratio of memory access to computation is low. 

The Warp State Statistics chart in nsight-compute shows the impact of these problems: on average per instruction executed a warp spends 31 cycles stalled on shared memory throttles (MIO), 15 cycles stalled on barrier waits and 11 stalled on long scoreboard (global load) dependencies.

![kernel-1-warp-stats](/assets/images/kernel-1-warp-stats-1.png)

 We can also use the profiler to  query the count of `mma` instructions executed, all instructions executed and elapsed cycles for each SM sub-partition:
```bash
------------------------------------------- ----------- -------------
Metric Name                                 Metric Unit  Metric Value
------------------------------------------- ----------- -------------
sm__cycles_elapsed.avg                            cycle 11,787,459.33
sm__cycles_elapsed.max                            cycle    11,822,127
sm__cycles_elapsed.min                            cycle    11,739,016
sm__cycles_elapsed.sum                            cycle 1,508,794,794
smsp__inst_executed.avg                            inst     2,051,328
smsp__inst_executed.max                            inst     2,067,354
smsp__inst_executed.min                            inst     2,035,302
smsp__inst_executed.sum                            inst 1,050,279,936
smsp__inst_executed_pipe_tensor_op_hmma.avg        inst        65,536
smsp__inst_executed_pipe_tensor_op_hmma.max        inst        66,048
smsp__inst_executed_pipe_tensor_op_hmma.min        inst        65,024
smsp__inst_executed_pipe_tensor_op_hmma.sum        inst    33,554,432
------------------------------------------- ----------- -------------
```
The total number of `mma` instructions is 33,554,432 as calculated earlier, with 65,536 being computed on each Tensor Core. The number of cycles elapsed per `mma` was 11,787,459 / 65,536 = 179.9, we are far from the 32 cycles best case. Another useful statistic is the ratio of `mma` instructions to total instructions which is `65,536 / 2,051,328 = 0.03`, so in Kernel 1 for each `mma` instruction we perform around 30 other instructions to load data, compute addresses etc. 

These three problems with Kernel 1 will be addressed in Kernel 2: Point 1 by using vectorized and coalesced loads, Point 2 by using a permuted shared memory layout and Point 3 as each warp will compute multiple output tiles. 

To isolate the impact made just by tiling vs the other changes, we add 2x tiling in the M and N dimensions in Kernel 1b. In this kernel each warp executes 4 `mma` instructions meaning each thread block computes a 64x64 output tile. This reduces execution time to 2.47 ms, increasing throuput to 55.6 TFLOP/S, 36% of cuBLAS performance.

### Aside: Floating Point Accuracy
NVIDIA does not fully document the exact numerical behavior of the Tensor Core `mma` instruction. The PTX ISA states: 
![mma-numeric](/assets/images/mma-numeric.png)
Getting into these details is not the focus of this post, but one example of rounding error is worth noting. Kernel 1 accumulates the results of the main loop over K directly in `dReg` meaning at each iteration the accumulation `dReg = dReg + aReg * bReg` happens within the `mma` operation, which can cause loss of precision if `dReg` is large compared to `aReg * bReg`.

When testing correctness of the implentation I initialize inputs with `U[0,1)` values. This means `dReg` grows monotonically as we loop over K, and performing the acculation directly in the `mma` operation causes round off such that the result using `mma` is consistently lower than a reference implmentation using `fp16/fp32` operations on CUDA cores (relative difference around 1e-5). This issue can be avoided by instead perfoming an mma without accumulation, and accumulating the results outside, i.e.
```c++
float cReg[4] = {0.};
mma_m16n8k16(aPtr, bPtr, cReg, dReg);
float4 *dRegPtr = reinterpret_cast<float4 *>(dReg);
dRegAcc.x += dRegPtr->x;
dRegAcc.y += dRegPtr->y;
dRegAcc.z += dRegPtr->z;
dRegAcc.w += dRegPtr->w;
```
This incurs a performance penalty of around `50us`, reduces the difference by two orders of magnitude and centers it. This is demonstrated in Kernel 1b but I have not included in the subsequent kernels as whether it's requried or not will depend on input data and desired accuracy level. Detailed investigation into the numerical behavior of Tensor Cores in general can be found in [^4].

## Kernel 2: Vectorized Loads & Permuted Shared Memory Layout
### Notes
Random things I have discovered so I don't forget
- Bank conflict count from ncu is misleading, if shared mem ideal == actual then no bank conflct as detailed here <https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/3>. This only seems to happen on kernel 3. 
- ncu fixes clock to base so times slower than nsys and cudaEvent which don't do this: <https://forums.developer.nvidia.com/t/nsight-compute-clock-speed-during-profiling/208646>.
- Using cp.async introduces other bank conflicts, but they show up in the source page but not in the details page, i.e. reverse of what is descibed in the link above.

## Kernel 2: Real Final version.v1.v1 (2)  
In this kernel we use some of the techniques (vectorized loads and permuted shared memory layout) discussed in the GTC 2020 CUTLASS presentation[^3] to resolve the performance issues of Kernel 1. Diagrams in this and subsequent sections are taken from that presentation. 

Throughout this kernel we operate on `uint4` 128b vectors containing 8 consecutive `fp16` elements in the K dimension of A and B. Working with 128b vectors is natural when using Tensor Cores as the fundamental Tensor Core operation is an 8 by 8 by 128b matrix multiply, i.e. each 128b vector forms one row of the input matrices. Using 128b vectors also means we can vectorize memory operations. 

The main loop of the kernel is shown below:

```c++
// column index when storing to shared memory
int storeCol = (laneID % 8) ^ (laneID / 8);

// indices when loading from shared memory
int loadRowA = (laneID % 16) / 2;
int loadColA = (laneID / 16 + 4 * (laneID % 2)) ^ (loadRowA % 4);
int loadRowB = (laneID % 8) / 2;
int loadColB = (laneID / 8 + 4 * (laneID % 2)) ^ (loadRowB % 4);

for (int k = 0; k < K/8; k += 4) {
  As[warpID*4 + laneID/8][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
  Bs[warpID*4 + laneID/8][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
  __syncthreads();

  // loop over the warps two (M=16/N=8, K=4) tiles of A and B
  for (int m = 0; m < 2; m++) {
    int mTile = m * 8;
    for (int n = 0; n < 2; n++) {
      int nTile = n * 4;
      load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + loadColA));
      load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + loadColB));
      mma_m16n8k16(aReg, bReg, dReg[m][n]);
      load_matrix_x4(aReg, (As[mWarp + mTile + loadRowA] + (loadColA^2)));
      load_matrix_x2(bReg, (Bs[nWarp + nTile + loadRowB] + (loadColB^2)));
      mma_m16n8k16(aReg, bReg, dReg[m][n]);
    }
  }
  __syncthreads();
}
```
Looking first at the load from global to shared, we keep the 16x16 thread block from Kernel 1. Each thread block loads tiles of A and B of shape (M/N=64, K=4) `uint4` values from global memory to shared memory in a K-major fashion (i.e. row-major for A and column-major for B), with consecutive threads loading consecutive `uint4` values in the K-dimension. To coalesce these loads, A is stored row-major in global memory while B is stored column-major. 

At the warp level, each warp loads a tile of shape (M/N=8, K=4). This tile is stored in a `uint4` shared memory array of shape (4, 8) with two K=4 row/column slices stored per shared memory row. This shared memory shape is used as shared memory has 32 banks which are each 4 bytes wide, hence a row of 8 `uint4` values spans the 32 shared memory banks. 

To avoid bank conflicts, threads which are part of the same memory request must not access addresses which map to the same bank. When each thread requests a 16B (128b) value, the warp level 512B request is split into 4 phases each consisting of 8 consecutive threads, as the max shared memory bandwidth is 32 banks * 4B = 128B. This means that it is sufficient to avoid bank conflicts within each phase of 8 threads, rather within the full warp of 32 threads.

When storing to shared memory, the column indices for each row are permuted by XORing them with the row index: `storeCol = (laneID % 8) ^ (laneID / 8)`. The store from global to shared would be bank conflict free without this permutation, but it is required to avoid bank conflicts when loading data to registers from shared memory. 

This diagram from [^3] illustrates how one warp loads from global to shared using the permuted layout:

![load-global-store-shared](/assets/images/load-global-store-shared.png)

Once data is loaded to shared memory, each warp computes a matmul on a (M=32, K=4) tile of A and a (N=16, K=4) tile of B. As the `mma` instruction computes a M=16, N=8, K=16 matmul we split these tiles into two (M=16, K=4) tiles of A / (N=8, K=4) tiles of B and compute their products in a nested loop. At the innermost level of this loop, we first load the k=0..1 subtiles of the current A and B tiles into registers and compute their product using the `mma` instruction. We then load the k=2..3 subtiles and perform a second `mma`. 

We use the `ldmatrix` PTX instruction to load these tiles from shared memory to registers. This warp-wide instruction loads 1, 2 or 4 8x128b matrices, each 8x128b matrix being distributed into one 32b register per thread in the fragment layout previously discussed. Each 128b row of these matrices is stored in one `uint4` vector in shared memory and each thread in the warp provides the address of one of these rows as described in the docs:

![ldmatrix-docs](/assets/images/ldmatrix-ptx-docs.png)

This means that to load a (M=16, k=0..1) subtile of `A`, we use the `.x4` variant of `ldmatrix`, with threads 0..15 providing the addresses of the elements with indices `m=0..15, k=0` and threads `16..31` providing the addresses of elements with indices `m=0..15, k=1`. Crucially, we permuted the layout of the tiles of A when storing to shared memory, and hence each thread needs to compute the address of its requried element in the permuted layout. 

Each (M=16, K=4) tile of A is stored in a 8 consecutive row subarray of the `As` shared memory array and each (N=8, K=4) tile of B is stored in a 4 consecutive row subarray of `Bs`. The `mWarp, nWarp` and `mTile, nTile` variables specify the start row of the subarrays of `As`, `Bs` for each warp / each iteration of the tile loop. Within each subarray the `loadRowA/B, loadColA/B` variables specify the location of the required element in the permuted layout. The following diagram from [^3] illustrates the mapping from elements required for a (M=16, K=4) tile of A to locations in shared memory:

![shared-register](/assets/images/shared-register.png)

The elements of the `k=0` slice of the subtile, loaded by threads `0..15` are shaded in blue. The elements loaded by threads `0..7` are all in distinct shared memory banks due to the permuted layout, as are those loaded by threads `8..15` and hence there are no bank conflicts. This is also true for the `k=1` slice which is shaded in green. If the permutation had not been applied, threads `0,2,4,6` would all access banks `0..3` and threads `1,3,5,7` would all access banks `16..19`, causing multiple bank conflicts.

The elements shaded in yellow/gray belong to the `k=2..3` slices, which are inputs to the second `mma`. The column indices for these slices can be computed efficiently from the column indices of the `k=0..1` slices by applying `xor 2` to those indices.

Loading the k=0..1 and k=2..3 subtiles of `B` is similar except that as the subltile dimension is (N=8, k=0..1) there are only 16 128b matrix rows to load. Hence we use the `.x2` `ldmatrix` which loads 2 8x128b matrices, usign only the addresses in threads `0..15`.

As with the `mma` instruction, we define helper functions `load_matrix_x4`, `load_matrix_x2` to wrap the inline PTX. Looking at `load_matrix_x4` as an example:
{% raw %}
```c++
__device__ void load_matrix_x4(unsigned *destReg, uint4 *srcAddr) {
  unsigned ptxSrcAddr = __cvta_generic_to_shared(srcAddr);
  asm volatile(
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"
      : "=r"(destReg[0]), "=r"(destReg[1]), "=r"(destReg[2]), "=r"(destReg[3])
      :  "r"(ptxSrcAddr)
      );
}
```
{% endraw %}
Two things to note
1. `__cvta_generic_to_shared` is a CUDA function that takes a standard C/C++ pointer, which is 64b, and converts to a 32b pointer as shared memory is a 32b address space
2. The `volatile` qualifier is needed for this instruction: without it the loads do not get synchronised properly and threads end up with incorrect data, which I discovered after much painful debugging.

One the main loop has finished, the output tile for each warp is accumulated in the output registers used for the `mma` instructions. There is a `stmatrix` instruction but this requires `sm_90` so is not available on Ada. We write directly from registers to global memory, it may be possible to optimize this by writing first to shared and then writing to global in a coalesced pattern but that requires more shared memory per threadblock which could reduce occupancy. I experimented with this but did not see a performance improvement.

### Performance
Kernel 2 has greatly increased performance. Execution time is 1060 us, a throughput of 129.7 TFLOP/S which is 81.1% cuBLAS and 78.5% of RTX 4090 peak performance. We can make one minor tweak to the kernel to improve performance further, currently we reload each tile of A for each tile of B, this reduces register usage but introduces redundant loads from shared memory to registers. 

In Kernel 2b we only load each tile of A once. This improves performance to 1030 us, 134.7 TFLOP/s, 87.3% of cuBLAS. The elapsed cycles per mma for Kernel 2b is 38, much closer to the minimum of 32. The ratio of total instrucitons to `mma` instructions is reduced from 31 for Kernel 1 to 3.9 for Kernel 2b.

The permuted shared memory layout should make these kernels bank-conflict free and we verify this for Kernel 2b:

![kernel-2b-conflict](/assets/images/kernel-2b-conflict.png)

Looking at the warp stats shows that the most frequent cause of stalls is now waiting for the Tensor Cores to be free - this is good!

![2b-warp-stats](/assets/images/kernel-2b-warp-stats-1.png)


There are still considerable number of barrier and long scoreboard stalls, which we'll address in Kernel 3 by introducing an n-stage pipeline from global to shared memory using the `cp.async` instruction.

## Kernel 3: N-stage global to shared pipeline
There are asyncronous copy APIs both in CUDA (`cuda::memcpy_async`) and PTX (`cp.async`). The `cuda::memcp_async` API does not support copying with a permuted layout and hence we use the PTX `cp.async` API. As before we deine a wrapper function for the inline PTX call:
{% raw %}
```c++
__device__ void cp_async(uint4 *dstAddr, const uint4 *srcAddr) {
  unsigned ptxDstAddr = __cvta_generic_to_shared(dstAddr);
  asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n"
      :: "r"(ptxDstAddr),
      "l"(srcAddr),
      "n"(16));
}
```
{% endraw %}
The final `"n"(16)` input is the number of bytes to copy, and needs to be a compile time constant.

We can then use this function to replace the global to shared load:
```c++
// Replace this
As[warpID*4 + laneID/8][storeCol] = globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
Bs[warpID*4 + laneID/8][storeCol] = globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4];
// With
cp_async(As[warpID*4 + laneID/8] + storeCol, globalTileA[(warpID*8 + laneID/4)*K/8 + k + laneID%4]);
cp_async(Bs[warpID*4 + laneID/8] + storeCol, globalTileB[(warpID*8 + laneID/4)*K/8 + k + laneID%4]);
asm volatile("cp.async.commit_group;\n" ::);
```
The `cp.async.commit_group` instruction groups these copies together in a `cp.async-group` which can later be waited on using `cp.async.wait_group`.

We now use `cp.async` to set up an n-stage pipeline from global to shared memory. We create circular buffers of size `N_STAGES` for A and B in shared memory. Before the main loop of the kernel we preload the first `N_STAGES - 1` stages into these shared memory buffers using `cp.async`:

```c++
__shared__ uint4 As[N_STAGES*32][8];
__shared__ uint4 Bs[N_STAGES*32][8];
// PRELUDE: load first (N_STAGES - 1) into shared memory
for (int nStage=0; nStage < N_STAGES - 1; nStage++) {
  int kStart = nStage * 4;
  aStorePtr = As + 32 * nStage;
  bStorePtr = Bs + 32 * nStage;
  cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
  cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
  asm volatile("cp.async.commit_group;\n" ::);
}
```
At the start of the main loop there are at most `N_STAGES-1` `cp.async` operations pending, this is an invariant that will be maintained at each loop iteration. We initialize shared memory load and store pointers to stages `0` and `N_STAGES-1` respectively and then wait for the first copy to complete, i.e. until there are at most `N_STAGES-2` cp.async operations pending. Note that a `__syncthreads` is required after `wait_group` as `wait_group` just synchronizes copy operations within each thread, not across threads.

```c++
//  MAIN LOOP OVER K BLOCKS
for (int nStage=0; nStage < K/32; nStage++) {
  int kStart = (N_STAGES-1+nStage) * 4;
  aStorePtr = As + 32 * ((nStage + N_STAGES-1) % N_STAGES);
  bStorePtr = Bs + 32 * ((nStage + N_STAGES-1) % N_STAGES);
  aLoadPtr = As + 32 * (nStage % N_STAGES);
  bLoadPtr = Bs + 32 * (nStage % N_STAGES);
  
  asm volatile("cp.async.wait_group %0;\n" :: "n"(N_STAGES-2));
  __syncthreads();

  // Preload the fragments for k=0..1, k=2..3 for both A/B tiles 
  for (int m=0; m<2; m++) {
    load_matrix_x4(aReg[m]    , aLoadPtr[m*8 + warpOffsetA + loadRowA] + loadColA);
    load_matrix_x4(aReg[m] + 4, aLoadPtr[m*8 + warpOffsetA + loadRowA] + (loadColA^2));
  }
  for (int n=0; n<2; n++) {
    load_matrix_x2(bReg[n]   , bLoadPtr[n*4 + warpOffsetB + loadRowB] + loadColB);
    load_matrix_x2(bReg[n]+ 2, bLoadPtr[n*4 + warpOffsetB + loadRowB] + (loadColB^2));
  }

  // Start next cp.async: on last N_STAGES-1 iterations the results of 
  // these copies are not used. The copies are done solely to allow
  // us to keep the argument to `wait_group` fixed at N_STAGES-2
  kStart = (kStart > 512-4) ? 512-4 : kStart;
  cp_async(aStorePtr[storeRow] + storeCol, aGlobalAddress + kStart);
  cp_async(bStorePtr[storeRow] + storeCol, bGlobalAddress + kStart);
  asm volatile("cp.async.commit_group;\n" ::);

  // Compute the mmas
  for (int m=0; m<2; m++) {
    for (int n=0; n<2; n++) {
      mma_m16n8k16(aReg[m]    , bReg[n]    , dReg[m][n]);
      mma_m16n8k16(aReg[m] + 4, bReg[n] + 2, dReg[m][n]);
    }
  }
}
```
Next we load the current shared memory stage to registers. In this kernel we preload the entire `M/N=64, K=4` tile into registers, requiring 16 registers for `A` and 8 for `B`. The extra shared memory required by the `N_STAGES` shared memory buffers is the occupancy bottleneck so using these extra registers makes sense to parallelize the loads as much as possible. After the starting the loads to registers, we submit the next `cp.async` instruction, and finally we perform the `mma` instructions and increment the load and store pointers modulo N_STAGES.

As `N_STAGES-1` K blocks were loaded before the main loop, on the last `N_STAGES-1` iterations through the main loop we don't need to load any more data from global memory. However the argmument to `cp.async.wait_group` needs to be a compile time constant and submitting superfluous copies is a hacky way to keep the argument to `wait_group` fixed at `N_STAGES-2`. Without these copies the kernel would be incorrect unless we decreased this argument on each of the last `N_STAGES-1` iterations.

### Performance
Sadly after all that effort Kernel 3 is a very minor improvement over Kernel 2b. Setting `N_STAGES=3` seems to give the best performance, `N_STAGES=4` is roughly the same and higher has worse performance. For `N_STAGES=3`, the execution time is 1000 us, giving 137.4 TFLOP/s, 89% cuBLAS, 83.2% 4090 peak performance, with `N_STAGES=3`. Looking at the warp state stats shows that stalls are lower:

![3-warp-stats](/assets/images/kernel-3-warp-stats.png)

but this is is partially due to reduced occupancy: Kernel 2b has 32 warps per SM while Kernel 3 has 24 due to the extra shared memory requirements.

As stalls due to barrier synchronization is still high, a reasonable optimization is to try increasing the work each warp does within a main loop iteration. We do this in Kernel 3b by increasing the tiling in the M/N dimensions from 2 to 4. This doubles the threadblock tile size to `(M/N=128, K=4)` meaning that each warp performs 4x4x2=32 `mma` instructions per main loop iteration. 

Kernel 3b has an execution time of 890 us, giving throughput of 154.4 TFLOP/s, 100% cuBLAS, 93.5% of RTX 4090 peak performance. Looking at the warp state stats shows that the vast majority of stalls are now due to waiting for Tensor Cores, in fact each warp now waits on average 36 cycles for a Tenor Core to be available:

![3b-warp-stats](/assets/images/kernel-3b-warp-stats.png)

The ratio of elapsed cycles to mma instructions for Kernel 3b is 34.2, consistent with the figure of 93.5% peak performance. 

Surprisingly nsight-compute shows the Tensor Core utilization as only 47.3% so what is going on?

![tc-util](/assets/images/tensor-core-util.png)

It seems that nsight uses a fixed latency of 16 cycles when computing `smsp__pipe_tensor_op_hmma_cycles_active` as the metric is consistently 16x `smsp__inst_executed_pipe_tensor_op_hmma`. I think this is an error, the latency should be 32 for the `m16n8k16` `mma` instruction. 

One final thing I noticed is that both Kernels 3 & 3b have bank conflicts, 3b nsight shows:

![kernel-3b-conflict](/assets/images/kernel-3b-conflict.png)

Confusingly in this view (Memory Tables) the conflicts appear only in the shared loads, whereas in the source metrics they appear both when copying from global to shared and when loading from shared to registers. The shared loads in particular use the same `ldmatrix` instruction as in Kernel 2, so I'm notsure how moving to `cp.async` introduces a conflict there. 

It's possible these conflicts are not real, nsight-compute reports erroneous conflicts in some cases as described [here](https://forums.developer.nvidia.com/t/shared-memory-bank-conflicts-and-nsight-metric/115731/12). I need to look into this further and will update the post if/when I find out what's going on here.

## Conclusion
We've gone from a naive implementation with correspondingly poor performance, to a kernel that is on par with cuBLAS, at least for this extremely specific problem formulation. In the process we've gained an understanding.

While that's a nice bonus, the real goal here was to understand the components of a high performance Tensor Core matrix multiply kernel and (at least for me) that's been a success. 


## Code 
Code is available here


### Remaing stuff is older version of Kernel 2 description - some still needs to be incorporated into current version

### High level overview
We keep the threadblock dimensions at blockDim.x=16, blockDim.y=16 so we have 256 threads per block. At the start of each main loop iteration we load tiles of A and B using vectorized 128b loads, each thread loading one uint4 128b vector containing eight consecutive fp16 values in the `K` dimension of `A` and `B`. At the warp level, we load `uint4` tiles of shape `(M/N=8, K=4)` of `A` and `B`, which correspond to `fp16` tiles of shape `(M/N=8, K=32)`. We store `A` row-major and `B` column-major in global memory, which means that per warp we load 8 rows/columns each containing 4 consecutive `uint4` values, which results in eight 64B memory transactions. Each transaction reads two 32B sectors out of a 128B cache line which contains four sectors in total. We have eight warps per threadblock so the threadblock tile dimensions (in terms of `uint4` vectors) are `(M/N=64, K=4)`. The code for the load instructions is
```c++
  __shared__ uint4 As[32][8];
  __shared__ uint4 Bs[32][8];
  for (int kStart=0; kStart < (K+K_BLOCK-1)/K_BLOCK; kStart++) {
    As[warpID*4 + laneID/8][storeCol] = globalA[(blockIdx.y*64 + warpID*8 + laneID/4)*K/8 + 4*kStart + laneID%4];
    Bs[warpID*4 + laneID/8][storeCol] = globalB[(blockIdx.x*64 + warpID*8 + laneID/4)*K/8 + 4*kStart + laneID%4];
    __syncthreads();
```
TODO: Explain that two of the K=4 wide rows get concenated into one row of shared memory.

Once the data is loaded into shared memory, we need load from shared memory to registers and carry out the `mma` instructions. We have tiles of `A` and `B` of shape `(M/N=64, K=4)` which equates to 32 `fp16` elements in the K dimension. As we are using the `m16n16k8` `mma` instructions, each warp needs to carry out eight `mma` instructions: two in each of the `M, N, K` dimensions. We use the same `(2, 4)` layout of warps in the output tile, but within this layout each warp is tiled twice in both the `M` and `N` directions.

```c++
// arrangement of warps in output tile
(warp_0 | warp_0 | warp_1 | warp_1 | warp_2 | warp_2 | warp_3 | warp_3) 
(warp_0 | warp_0 | warp_1 | warp_1 | warp_2 | warp_2 | warp_3 | warp_3) 
(warp_4 | warp_4 | warp_5 | warp_5 | warp_6 | warp_6 | warp_7 | warp_7) 
(warp_4 | warp_4 | warp_5 | warp_5 | warp_6 | warp_6 | warp_7 | warp_7) 
```

Within each of these warp subtiles, we need to compute two `mma` instructions in the `K` dimension, as we have loaded `K=32` wide tile of `A` and `B` and the `mma` instruction acts on `K=16` wide tiles. 


### Coalesced Access Notes
From [CUDA Programming Guide 5.3.2](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#device-memory-accesses), global memory is accessed via 32, 64 or 128 byte memory transactions. Each `uint4` is 16 bytes, and per warp we read 8 rows, each containing 4 consecutive `uint4` values. So this is 8, 64 byte access in total? Sector is 32 bytes, and a L1/2 cache line is four sectors. One question I have is currently we read 8x two sectors, is it possible to improve this to 4x cache line?

### Vectorized Loads
Using [vectorized loads](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/) is important to maximize achieved bandwidth. Rather than having a thread load one `fp16` element, we load one `unit4` 128b vector (containing 8 `fp16` values) per thread. 

We keep the threadblock dimensions 16x16 which means that we can load 256 uint4 / 2048 fp16 values into shared memory in one threadblock wide load, which we use to load (M/N=64, K=32) tiles of A and B. To make this load coalesced we store A in row-major format and B in column-major format in global memory.

We use the same (2, 4) layout of warps as in Kernel 1, meaning that as each warp computes a `m16n8k16` matmul, threadblock wide we compute a `m32k16n32` matmul. As the threadblock tiles are shape `(64, 32)`, we'll perform warp tiling, with each warp computing matmuls for 2 subtiles in each of the `M, N, K` directions.

### Permuted shmem layout
Avoiding shared memory bank conflicts is crucial for performance. CUTLASS achieves this by using a permuted layout of matrix elements in shared memory, and we are going to use this permuted layout in this kernel. Shared memory has 32 banks which are each 4 bytes wide, hence 8 `unit4` values span the 32 shared memory banks. Normally to avoid bank conflicts we need to ensure within a warp no two threads access different addresses in the same bank, however when accessing 16 byte words the warp is split into 4 consecutive phases of 8 threads and it is sufficient to avoid bank conflicts between the threads in the same phase. 

We store the `(64, 32)` tiles of `A` and `B` in `uint4` shared memory arrays of shape (32, 8), so each row of the shared memory array spans shared memory and contains two row/column slices of `A`/`B`. When storing the elements to shared memory, the column used in the store is permuted by applying an XOR on the column index `storeCol = (laneID % 8) ^ (laneID / 8)`, as shown in this diagram taken from [^3]:

![load-global-store-shared](/assets/images/load-global-store-shared.png)

The store from global to shared would be conflict free without this permutation, but it is required to avoid bank conflicts when loading from shared memory to registers. To see why, lets look at loading a tile of `A` from shared memory to registers. 

In each inner iteration of the main loop we will perform two `mma` instructions, one for `K=0,1` and one for `K=2,3` (as `K` here is `128b` wide, these are `K=16` in the original `half` precision elements).

For each `mma` instruction we load one `(M=16, K=2)` subtile of `A`. The mapping from `threadID` to loaded element is transposed from the mapping used when storing from global to shared: for the first `mma` threads 0 to 15 load the `K=0` slice of the threadblock tile, and theads 16 to 31 the `K=1` slice, and for the second `mma` threads 0 to 15 load `K=2` and threads 16 to 31 load `K=3`. The position of these elements in the permuted shared memory array is as shown in the following diagram:

![shared-register](/assets/images/shared-register.png)

Notice that the eight blue shaded (K=0) 128b vectors loaded by threads `0..7` are all in different shared memory banks due to the permutation that was applied when storing them and hence there are no bank conflicts from the first phase of the shared memory load. If the permuation had not been applied threads `0,2,4,6` would all access elements in banks `0..3`, and threads `1,3,5,7` would access elements in banks `16..19`, causing bank conflicts. Similary there are no conflicts between  threads each of the remaing three phases.

The code to compute the indices loaded by each thread is as follows:

INSERT CODE

The yellow and grey shaded `uint4` vectors correspond to `K=3` and `K=4` respectively. These are inputs to the second `mma` performed in the `K` dimension. There is a neat trick that is applied to get their column index from the `K=0/1` indices: simply apply `xor 2` to those indices.  

Loading `B` is similar except we load a `(N=8, K=16)` tile for each `mma`.

### References
[^1]: [GTC 2019 Programming Tensor Cores: Navtive Volta Tensor Cores With CUTLASS](https://developer.download.nvidia.com/video/gputechconf/gtc/2019/presentation/s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf)

[^2]: Originally FLOPS stood for Floating point Operations Per Second. However In deep learning it is also used a measure of quantity i.e. to mean Floating point Operations. To prevent confusion I am using FLOP/s for rates and FLOP for quantities, as suggested [here](https://blog.heim.xyz/flop-for-quantity-flop-s-for-performance).

[^3]: [GTC 2020 Developing CUDA Kernels to Push Tensor Cores to the Absolute Limite on NVIDIA A100](https://www.nvidia.com/en-us/on-demand/session/gtcsj20-s21745)

[^4]: [Numerical Behavior of NVIDIA Tensor Cores](https://eprints.maths.manchester.ac.uk/2774/1/fhmp20.pdf)

[^5]: [Why would code run 1.7x faster when run with nvprof than without](https://forums.developer.nvidia.com/t/why-would-code-run-1-7x-faster-when-run-with-nvprof-than-without/56406/7)

[^6]: [Ada Architecture White Paper](https://images.nvidia.com/aem-dam/Solutions/Data-Center/l4/nvidia-ada-gpu-architecture-whitepaper-v2.1.pdf)

[^7]: [Benchmarking and Dissecting the Nvidia Hopper GPU Architecture](https://arxiv.org/pdf/2402.13499v1)

[^8]: [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
