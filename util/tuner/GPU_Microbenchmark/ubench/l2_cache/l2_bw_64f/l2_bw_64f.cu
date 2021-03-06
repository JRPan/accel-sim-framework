// This code is a modification of L2 cache benchmark from
//"Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking":
// https://arxiv.org/pdf/1804.06826.pdf

// This benchmark measures the maximum read bandwidth of L2 cache for 64 bit
// Compile this file using the following command to disable L1 cache:
//    nvcc -Xptxas -dlcm=cg -Xptxas -dscm=wt l2_bw.cu

#include <algorithm>
#include <assert.h>
#include <cuda.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "../../../hw_def/hw_def.h"

#define REPEAT_TIMES 2048

/*
L2 cache is warmed up by loading posArray and adding sink
Start timing after warming up
Load posArray and add sink to generate read traffic
Repeat the previous step while offsetting posArray by one each iteration
Stop timing and store data
*/

__global__ void l2_bw(uint32_t *startClk, uint32_t *stopClk, double *dsink,
                      double *posArray, unsigned ARRAY_SIZE) {
  // block and thread index
  uint32_t tid = threadIdx.x;
  uint32_t bid = blockIdx.x;
  uint32_t uid = bid * blockDim.x + tid;

  // a register to avoid compiler optimization
  double sink = 0;

  // warm up l2 cache
  for (uint32_t i = uid; i < ARRAY_SIZE; i += blockDim.x * gridDim.x) {
    double *ptr = posArray + i;
    // every warp loads all data in l2 cache
    // use cg modifier to cache the load in L2 and bypass L1
    asm volatile("{\t\n"
                 ".reg .f64 data;\n\t"
                 "ld.global.cg.f64 data, [%1];\n\t"
                 "add.f64 %0, data, %0;\n\t"
                 "}"
                 : "+d"(sink)
                 : "l"(ptr)
                 : "memory");
  }

  asm volatile("bar.sync 0;");

  // start timing
  uint32_t start = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(start)::"memory");

  // benchmark starts
  // load data from l2 cache and accumulate,
  for (uint32_t i = 0; i < REPEAT_TIMES; i++) {
    double *ptr = posArray + (i * warpSize) + uid;
    asm volatile("{\t\n"
                 ".reg .f64 data;\n\t"
                 "ld.global.cg.f64 data, [%1];\n\t"
                 "add.f64 %0, data, %0;\n\t"
                 "}"
                 : "+d"(sink)
                 : "l"(ptr)
                 : "memory");
  }
  asm volatile("bar.sync 0;");

  // stop timing
  uint32_t stop = 0;
  asm volatile("mov.u32 %0, %%clock;" : "=r"(stop)::"memory");

  // store the result
  startClk[bid * blockDim.x + tid] = start;
  stopClk[bid * blockDim.x + tid] = stop;
  dsink[bid * blockDim.x + tid] = sink;
}

int main() {

  intilizeDeviceProp(0);

  unsigned ARRAY_SIZE = TOTAL_THREADS + REPEAT_TIMES * WARP_SIZE;
  // Array size must not exceed L2 size
  assert(ARRAY_SIZE * sizeof(double) < L2_SIZE);

  uint32_t *startClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));
  uint32_t *stopClk = (uint32_t *)malloc(TOTAL_THREADS * sizeof(uint32_t));

  double *posArray = (double *)malloc(ARRAY_SIZE * sizeof(double));
  double *dsink = (double *)malloc(TOTAL_THREADS * sizeof(double));

  double *posArray_g;
  double *dsink_g;
  uint32_t *startClk_g;
  uint32_t *stopClk_g;

  for (int i = 0; i < ARRAY_SIZE; i++)
    posArray[i] = (double)i;

  gpuErrchk(cudaMalloc(&posArray_g, ARRAY_SIZE * sizeof(double)));
  gpuErrchk(cudaMalloc(&dsink_g, TOTAL_THREADS * sizeof(double)));
  gpuErrchk(cudaMalloc(&startClk_g, TOTAL_THREADS * sizeof(uint32_t)));
  gpuErrchk(cudaMalloc(&stopClk_g, TOTAL_THREADS * sizeof(uint32_t)));

  gpuErrchk(cudaMemcpy(posArray_g, posArray, ARRAY_SIZE * sizeof(double),
                       cudaMemcpyHostToDevice));

  l2_bw<<<BLOCKS_NUM, THREADS_PER_BLOCK>>>(startClk_g, stopClk_g, dsink_g,
                                           posArray_g, ARRAY_SIZE);
  gpuErrchk(cudaPeekAtLastError());

  gpuErrchk(cudaMemcpy(startClk, startClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(stopClk, stopClk_g, TOTAL_THREADS * sizeof(uint32_t),
                       cudaMemcpyDeviceToHost));
  gpuErrchk(cudaMemcpy(dsink, dsink_g, TOTAL_THREADS * sizeof(double),
                       cudaMemcpyDeviceToHost));

  float bw, BW;
  unsigned long long data =
      (unsigned long long)TOTAL_THREADS * REPEAT_TIMES * sizeof(double);
  uint64_t total_time = stopClk[0] - startClk[0];
  bw = (float)(data) / ((float)(total_time));
  BW = bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "L2 bandwidth = " << bw << "(byte/clk), " << BW << "(GB/s)\n";
  float max_bw = get_num_channels(MEM_BITWIDTH, DRAM_MODEL) *
                 L2_BANKS_PER_MEM_CHANNEL * L2_BANK_WIDTH_in_BYTE;
  BW = max_bw * CLK_FREQUENCY * 1000000 / 1024 / 1024 / 1024;
  std::cout << "Max Theortical L2 bandwidth = " << max_bw << "(byte/clk), "
            << BW << "(GB/s)\n";
  std::cout << "L2 BW achievable = " << (bw / max_bw) * 100 << "%\n";
  std::cout << "Total Clk number = " << total_time << "\n";
  return 1;
}
