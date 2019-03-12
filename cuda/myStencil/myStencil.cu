#include <stdio.h>
#include <sys/time.h>

// For the CUDA runtime routines (prefixed with "cuda_")
// /usr/local/cuda-9.0/bin/nvcc -I../../common/inc --ptx myStencil.cu
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define THREADS 512
#define O 1
#define K 1
#define WARP_SIZE 32
#define RC

#ifndef RC
__global__ void one_stencil (const int *A, int *B, int sizeOfA)
{
    __shared__ int s[THREADS*O+2*K];
    
    // Id of thread in the block.
    int localId = threadIdx.x;

    // The first index of output element computed by this block.
    int startOfBlock = blockIdx.x * blockDim.x * O;

    // The Id of the thread in the scope of the grid.
    int globalId = localId + startOfBlock;

    if (globalId >= sizeOfA)
        return;

    // Fetching into shared memory.
    for(int i=0; i<O; i++)
      if(blockDim.x*i + globalId < sizeOfA) {
        s[blockDim.x*i + localId] = A[blockDim.x*i + globalId];
      }
    if (localId < 2*K && blockDim.x*O + globalId < sizeOfA) {
        s[blockDim.x*O + localId] =  A[blockDim.x*O + globalId];
    }

    // We must sync before reading from shared memory.
    __syncthreads();

    // Each thread computes a single output.
    for(int o=0; o<O; o++) {
    if (globalId + blockDim.x*o < sizeOfA - 2*K) {
       int acc = 0;
       for(int i=0; i < 2*K+1; i++)
         acc += s[blockDim.x*o + localId + i];;
        B[blockDim.x*o + globalId] = acc / (2*K+1);
    }
    }
}
#else
__global__ void one_stencil (int *A, int *B, int sizeOfA)
{
    // Declaring local register cache.
    int rc[O+1];

    // Id of thread in the warp.
    int localId = threadIdx.x % WARP_SIZE;

    // The first index of output element computed by this warp.
    int startOfWarp = (blockIdx.x * blockDim.x + WARP_SIZE*(threadIdx.x / WARP_SIZE)) * O;

    // The Id of the thread in the scope of the grid.
    int globalId = localId + startOfWarp;

    if (globalId >= sizeOfA)
        return;

    // Fetching into shared memory.
    for(int i=0; i<O; i++)
      if(WARP_SIZE*i + globalId < sizeOfA)
        rc[i] = A[WARP_SIZE*i + globalId];	
    if (localId < 2*K && WARP_SIZE*O + globalId < sizeOfA)
    {
        rc[O] =  A[WARP_SIZE*O + globalId];
    }

    // Each thread computes a single output.
    unsigned mask = __activemask();
    for(int o=0; o<O; o++) {
      int ac = 0;
      int toShare = rc[o];	

      for (int i = 0 ; i < 2*K+1 ; ++i)
      {
        // Threads decide what value will be published in the following access.
        if (localId < i)
            toShare = rc[o+1];

        // Accessing register cache.
        ac += __shfl_sync(mask, toShare, (localId + i) % WARP_SIZE);
      }

      if (globalId + o*WARP_SIZE  < sizeOfA - 2*K)
        B[globalId + o*WARP_SIZE] = ac/(2*K+1);

    }
}
#endif

/**
 * Host main routine
 */
int
main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 50000000;
    size_t size = numElements * sizeof(int);
    printf("[Vector addition of %d elements]\n", numElements);

    // Allocate the host input vector A
    int *h_A = (int *)malloc(size);

    // Allocate the host input vector B
    int *h_B = (int *)malloc(size);
    
    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the host input vectors
    for (int i = 0; i < numElements; ++i)
    {
        h_A[i] = rand(); // /RAND_MAX;
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

    // Copy the host input vectors A and B in host memory to the device input vectors in
    // device memory
    printf("Copy input data from the host memory to the CUDA device\n");
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Vector Add CUDA Kernel
    struct timeval t0, t1;
    int threadsPerBlock = THREADS;
    int blocksPerGrid =(numElements + threadsPerBlock*O - 1) / (threadsPerBlock*O);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    one_stencil<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, numElements);
    cudaDeviceSynchronize();
    gettimeofday(&t0, NULL); 
    one_stencil<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, numElements);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    
    err = cudaGetLastError();
    long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    printf("execution time: %ld us\n", elapsed);

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

    // Verify that the result vector is correct
    for (int i = 0; i < numElements-2*K; ++i)
    {
	int acc = 0;
        for(int j=0; j<2*K+1; j++)
	  acc += h_A[i+j];
        if (acc/(2*K+1) != h_B[i])
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
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

    // Free host memory
    free(h_A);
    free(h_B);

    printf("Done\n");
    return 0;
}
