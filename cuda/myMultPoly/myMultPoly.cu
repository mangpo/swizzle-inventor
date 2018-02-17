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
#define WARP_SIZE 32
#define Y_THREADS (512/WARP_SIZE)

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#define RC
//#define CHECK

#ifdef CHECK
#define ROWS 1000
#define ITER 1
#else
#define ROWS 1000000
#define ITER 1
#endif

// 18722 vs 9571 (95% faster)
__global__ void multiply_shmem (const int *A, const int *B, int *C, int N)
{
  __shared__ int sA[Y_THREADS][WARP_SIZE];
  __shared__ int sB[Y_THREADS][WARP_SIZE];
  int output = 0;
  int lindex = threadIdx.x & (WARP_SIZE - 1);
  int gy = blockIdx.y * blockDim.y + threadIdx.y;
  int ly = threadIdx.y;

  sA[ly][lindex] = A[gy*WARP_SIZE + lindex];
  sB[ly][lindex] = B[gy*WARP_SIZE + lindex];
  __syncthreads();

  for(int i=0; i<=lindex; i++) {
    int a = sA[ly][i];
    int b = sB[ly][lindex-i];
    output ^= a&b;
  }
  C[gy*WARP_SIZE*2 + lindex] = output;
  output = 0;
  for(int i=lindex+1; i<N; i++) {
    int a = sA[ly][i];
    int b = sB[ly][N-1+lindex-i];
    output ^= a&b;
  }
  C[gy*WARP_SIZE*2 + lindex + N] = output;
}

__global__ void multiply_rc (const int *A, const int *B, int *C, int N)
{
  int a_cached, b_cached;
  int output = 0;
  int lindex = threadIdx.x & (WARP_SIZE - 1);
  int gy = blockIdx.y * blockDim.y + threadIdx.y;

  a_cached = A[gy*WARP_SIZE + lindex];
  b_cached = B[gy*WARP_SIZE + lindex];

  unsigned mask = __activemask();

/*
  for(int i=0; i<N; i++) {
    int a = __shfl_sync(mask, a_cached, i);
    int b = __shfl_sync(mask, b_cached, lindex-i);
    if(i<=lindex) output ^= a&b;
  }
  C[gy*WARP_SIZE*2 + lindex] = output;
  output = 0;
  for(int i=0; i<N; i++) {
    int a = __shfl_sync(mask, a_cached, i);
    int b = __shfl_sync(mask, b_cached, N-1+lindex-i);
    if(i>lindex) output ^= a&b;
  }
  C[gy*WARP_SIZE*2 + lindex + N] = output;
  */

  int output2 = 0;
  for(int i=0; i<N; i++) {
    int a = __shfl_sync(mask, a_cached, i);
    int b = __shfl_sync(mask, b_cached, lindex-i);
    int b2 = __shfl_sync(mask, b_cached, N-1+lindex-i);
    if(i<=lindex) output ^= a&b;
    else output2 ^= a&b2;
  }
  C[gy*WARP_SIZE*2 + lindex] = output;
  C[gy*WARP_SIZE*2 + lindex + N] = output2;
}


/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 32; //50000;
    size_t size = numElements * ROWS * sizeof(int);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    int *h_A = (int *)malloc(size);

    // Allocate the host input vector B
    int *h_B = (int *)malloc(size);

    // Allocate the host input vector C
    int *h_C = (int *)malloc(2*size);
    
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements * ROWS; ++i)
    {
        h_A[i] = rand(); // /RAND_MAX;
        h_B[i] = rand(); // /RAND_MAX;
    }

    // Allocate the device input vector A
    int *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);

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

    // Allocate the device input vector C
    int *d_C = NULL;
    err = cudaMalloc((void **)&d_C, 2*size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef CHECK
    // Allocate the device input vector C2
    int *h_C2 = (int *)malloc(2*size);
    int *d_C2 = NULL;
    err = cudaMalloc((void **)&d_C2, 2*size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
#endif

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    struct timeval t0, t1;
    dim3 threadsPerBlock(WARP_SIZE, Y_THREADS);
    dim3 blocksPerGrid(1, (ROWS + Y_THREADS - 1)/Y_THREADS);
    printf("CUDA kernel launch: block = %d\n", (ROWS + Y_THREADS - 1)/Y_THREADS);
    
#ifndef RC
    multiply_shmem<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
#else
    multiply_rc<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
#endif
    cudaDeviceSynchronize();
    gettimeofday(&t0, NULL);
    for(int i=0; i<ITER; i++) {
#ifndef RC
      multiply_shmem<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
#else
      multiply_rc<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
#endif
      cudaDeviceSynchronize();
      }
    gettimeofday(&t1, NULL);
    err = cudaGetLastError();
    
    long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    printf("execution time: %ld us\n", elapsed);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch one_stencil #1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

#ifdef CHECK
    // Copy the device result vector in device memory to the host result vector
    // in host memory.
    printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_C, d_C, 2*size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    multiply_shmem<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C2, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch one_stencil #2 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(h_C2, d_C2, 2*size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C2 from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements * 2 * ROWS; ++i) {

      if(h_C[i] != h_C2[i])
      {
            fprintf(stderr, "Result verification failed at element %d: %d %d!\n", i, h_C2[i], h_C[i]);
            exit(EXIT_FAILURE);
      }
    }

    printf("Test PASSED\n");
#endif

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

    err = cudaFree(d_C);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    printf("Done\n");
    return 0;
}

