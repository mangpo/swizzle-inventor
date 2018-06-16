/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
// /usr/local/cuda-9.0/bin/nvcc -I../../common/inc --ptx myStencil.cu
#include <cuda_runtime.h>

#include <helper_cuda.h>
#define THREADS 256
#define WARP_SIZE 32
#define RC

#define M 6
#define c 2
#define a 3
#define b 16

struct unit {
  int x[M];
} __align__(8);

// 6: 24926 vs 19143 vs 30347
__global__ void r2c_naive (const struct unit *A, int *B, int sizeOfA)
{
    int globalId = threadIdx.x + blockDim.x * blockIdx.x;

    //if(globalId < sizeOfA) {
    int sum = 0;
    #pragma unroll
      for(int i=0; i<M; i++) {
    	sum += A[globalId].x[i];
      }
    //}
    B[globalId] = sum;
}

__global__ void r2c_bug (const int2 *A, int *B, int sizeOfA)
{

    //int warp_id = threadIdx.x/WARP_SIZE;
    //int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    int global_tid = (blockIdx.x * blockDim.x) + threadIdx.x;
    int warp_offset = M * (global_tid - j);

    struct unit x0,x1,x2;

    unsigned mask = __activemask();
    int sum = 0;

    #pragma unroll
    for(int i=0; i<M; i+=2) {
      int2 temp = A[warp_offset/2 + j + i*WARP_SIZE/2];
      x0.x[i] = temp.x;
      x0.x[i+1] = temp.y;
      //x0.x[i] = A[warp_offset + 2*j + i*WARP_SIZE];
      //x0.x[i+1] = A[warp_offset + 2*j + i*WARP_SIZE + 1];
    }

    int r20 = (3*j) & 31;
    int r22 = (3*j + 2) & 31;
    int r24 = (3*j + 1) & 31;

    //int r27 = (j - 3*(j/3));
    int r27 = j % 3;

    #pragma unroll
    for(int i=0; i<M; i++) {
      if((r27 & 1) == 0)
	x1.x[i] = x0.x[i];
      else
	x1.x[i] = x0.x[(i+2)%M];
    }
    #pragma unroll
    for(int i=0; i<M; i++) {
      if((r27 & 2) == 0)
	x2.x[i] = x1.x[i];
      else
	x2.x[i] = x1.x[(i-2+M)%M];
    }
    
    sum += __shfl_sync(mask, x2.x[0], r20);
    sum += __shfl_sync(mask, x2.x[1], r20);
    sum += __shfl_sync(mask, x2.x[2], r22);
    sum += __shfl_sync(mask, x2.x[3], r22);
    sum += __shfl_sync(mask, x2.x[4], r24);
    sum += __shfl_sync(mask, x2.x[5], r24);
    
    B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
}

__global__ void r2c_mod (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    __shared__ int x[THREADS][M];

    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
      for(int i=0; i<M; i++) {
	x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
      }

    
      for(int i=0; i<M; i++) {
	int inter = (i-j+1) % M;
        if(inter < 0) inter += M;
	int index = ((a * (i-j)) + inter/c) % M;
	if(index < 0) index += M;
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
        sum += __shfl_sync(mask, x[threadIdx.x][index], lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}


__global__ void r2c_lit (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    __shared__ int __align__(16) x[THREADS][M];
    __shared__ int perm[6]; // <-- using constant
    perm[0] = 0;
    perm[1] = 4;
    perm[2] = 1;
    perm[3] = 5;
    perm[4] = 2;
    perm[5] = 3;
 
    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
    #pragma unroll
      for(int i=0; i<M; i++) {
	x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
      }

      for(int i=0; i<M; i++) {
	int index = (i - j + 6*M) % M; //(modulo (- i j) struct-size))
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
        sum += __shfl_sync(mask, x[threadIdx.x][perm[index]], lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}


__global__ void r2c_lit_reg (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    __shared__ int __align__(16) x[THREADS][M];
    int perm = 0x325140;
 
    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
      for(int i=0; i<M; i++) {
	x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
      }

      for(int i=0; i<M; i++) {
	int index = (i - j + 6*M) % M; //(modulo (- i j) struct-size))
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
        sum += __shfl_sync(mask, x[threadIdx.x][(perm >> (index*4)) & 0x7], lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}

__global__ void r2c_lit_reg_struct (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    //__shared__ int __align__(16) x[THREADS][M];
    int perm = 0x325140;
    struct unit s;
 
    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
    #pragma unroll
      for(int i=0; i<M; i++) {
	//x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
	s.x[i] = A[warp_offset + j + i*WARP_SIZE];
      }

      #pragma unroll
      for(int i=0; i<M; i++) {
	int index = (i - j + 6*M) % M; //(modulo (- i j) struct-size))
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
        //sum += __shfl_sync(mask, x[threadIdx.x][(perm >> (index*4)) & 0x7], lane);
        sum += __shfl_sync(mask, s.x[(perm >> (index*4)) & 0x7], lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}



__global__ void r2c_lit_reg_strength (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    __shared__ int __align__(16) x[THREADS][M];
    int perm = 0x325140;
 
    unsigned mask = __activemask();
    int sum = 0;
    int lb = (j * M) % WARP_SIZE;
    int ub = (lb + M) % WARP_SIZE;
    int lane = (lb + j/b) % WARP_SIZE;

    //if(globalId < sizeOfA) {
      for(int i=0; i<M; i++) {
	x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
      }

      for(int i=0; i<M; i++) {
	int index = (i - j) % M;
        sum += __shfl_sync(mask, x[threadIdx.x][(perm >> (index*4)) & 0x7], lane);
	lane = (lane + 1) % WARP_SIZE;
	if(lane == ub) lane = lb;
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}

__global__ void r2c_lit_reg2 (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    //__shared__ int __align__(16) x[THREADS][M];
    int perm = 0x325140;
    int x0,x1,x2,x3,x4,x5;
 
    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
      x0 = A[warp_offset + j + 0*WARP_SIZE];
      x1 = A[warp_offset + j + 1*WARP_SIZE];
      x2 = A[warp_offset + j + 2*WARP_SIZE];
      x3 = A[warp_offset + j + 3*WARP_SIZE];
      x4 = A[warp_offset + j + 4*WARP_SIZE];
      x5 = A[warp_offset + j + 5*WARP_SIZE];

      #pragma unroll
      for(int i=0; i<M; i++) {
	int index = (i - j + 6*M) % M; //(modulo (- i j) struct-size))
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
	int in = (perm >> (index*4)) & 0x7;
	int x;
	if(in == 0) x = x0;
	else if(in == 1) x = x1;
	else if(in == 2) x = x2;
	else if(in == 3) x = x3;
	else if(in == 4) x = x4;
	else x = x5;
        sum += __shfl_sync(mask, x, lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}

__constant__ int myperm[6];
__global__ void r2c_lit_const (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    __shared__ int __align__(16) x[THREADS][M];
    /*
     __shared__ int perm[6]; // <-- using constant
    perm[0] = 0;
    perm[1] = 4;
    perm[2] = 1;
    perm[3] = 5;
    perm[4] = 2;
    perm[5] = 3;
    */
 
    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
      for(int i=0; i<M; i++) {
	x[threadIdx.x][i] = A[warp_offset + j + i*WARP_SIZE];
      }

      for(int i=0; i<M; i++) {
	int index = (i - j + 6*M) % M; //(modulo (- i j) struct-size))
        int lane = (((i + j/b) % M) + (j * M)) % WARP_SIZE;
        sum += __shfl_sync(mask, x[threadIdx.x][myperm[index]], lane);
      }

   B[blockIdx.x * blockDim.x + threadIdx.x] = sum;
    //}
}


/**
 * Host main routine
 */
int
main(void)
{
  // Define constants
  int perm[6];
  perm[0] = 0;
  perm[1] = 4;
  perm[2] = 1;
  perm[3] = 5;
  perm[4] = 2;
  perm[5] = 3;
  cudaMemcpyToSymbol(myperm, perm, 6*sizeof(int));
  
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    //int numElements = 64000000;
    int numElements = THREADS * 15 * 8 * 100;
    size_t size = numElements * sizeof(int);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    int *h_A = (int *)malloc(size*M);

    // Allocate the host input vector B
    int *h_B = (int *)malloc(size);
    int *h_B2 = (int *)malloc(size);
    
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_B2 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements * M; ++i)
    {
        h_A[i] = i; //rand(); // /RAND_MAX;
    }

    // Allocate the device input vector A
    int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size*M);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B
    int *d_B = NULL;
    err = cudaMalloc((void **)&d_B, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector B2
    int *d_B2 = NULL;
    err = cudaMalloc((void **)&d_B2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size*M, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    struct timeval t0, t1, t2;
    float time0, time1;
    cudaEvent_t start0, stop0, start1, stop1;
    cudaEventCreate(&start0);
    cudaEventCreate(&stop0);
    cudaEventCreate(&start1);
    cudaEventCreate(&stop1);

    int threadsPerBlock = THREADS;
    int blocksPerGrid = (numElements + threadsPerBlock - 1) / (threadsPerBlock);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // TODO: use CUDA event timer
    r2c_naive<<<blocksPerGrid, threadsPerBlock>>>((struct unit *) d_A, d_B, numElements);
    cudaDeviceSynchronize();
    
    cudaEventRecord(start0,0);
    gettimeofday(&t0, NULL);
    for(int i=0; i<10; i++)
      r2c_naive<<<blocksPerGrid, threadsPerBlock>>>((struct unit *) d_A, d_B, numElements);
    cudaEventRecord(stop0,0);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    
    cudaEventRecord(start1,0);
    for(int i=0; i<10; i++)
      r2c_bug<<<blocksPerGrid, threadsPerBlock>>>((int2 *) d_A, d_B2, numElements);
    cudaEventRecord(stop1,0);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    
    err = cudaGetLastError();
    long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    long elapsed2 = (t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec;
    printf("direct load:  %ld us\n", elapsed);
    printf("shuffle load: %ld us\n", elapsed2);

    cudaEventElapsedTime(&time0, start0, stop0);
    cudaEventElapsedTime(&time1, start1, stop1);
    cudaEventDestroy(start0);
    cudaEventDestroy(stop0);
    cudaEventDestroy(start1);
    cudaEventDestroy(stop1);
    
    printf("direct load (cuda):  %f ms\n", time0);
    printf("shuffle load (cuda): %f ms\n", time1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch one_stencil kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_B2, d_B2, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
	if(h_B[i] != h_B2[i]) {
	  printf("h_B[%d] = %d, h_B2[%d] = %d\n", i, h_B[i], i, h_B2[i]);
          exit(EXIT_FAILURE);
	}
    }

    printf("Test PASSED\n");

    // Free device global memory
    err = cudaFree(d_A);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_B);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaFree(d_B2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_B2);

    printf("Done\n");
    return 0;
}

