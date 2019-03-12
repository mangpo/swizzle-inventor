#include <stdio.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define THREADS 64
#define WARP_SIZE 32

#define M 3
#define c 1
#define a 3
#define b 32

struct unit {
  int x[M];
};

__global__ void r2c_naive (struct unit *A, int *B, int sizeOfA)
{
    int localId = threadIdx.x;
    int offset = blockIdx.x * blockDim.x;
    int globalId = offset + localId;

    //if(globalId < sizeOfA) {
    int sum = 0;
      for(int i=0; i<M; i++) {
    	sum += A[globalId].x[i];
      }
    //}
    B[globalId] = sum;
}

__global__ void r2c_mod (const int *A, int *B, int sizeOfA)
{

    int warp_id = threadIdx.x/WARP_SIZE;
    int warp_offset = M * ((blockIdx.x * blockDim.x) + (warp_id * WARP_SIZE));
    int j = threadIdx.x % WARP_SIZE;
    int x[M];

    unsigned mask = __activemask();
    int sum = 0;

    //if(globalId < sizeOfA) {
      for(int i=0; i<M; i++) {
	x[i] = A[warp_offset + j + i*WARP_SIZE];
      }

    
      for(int i=0; i<M; i++) {
	int index = (2*i + j) % M;
	int lane = (i + j*a) % WARP_SIZE;
        sum += __shfl_sync(mask, x[index], lane);
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
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = 256000000;
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
    int threadsPerBlock = THREADS;
    int blocksPerGrid =(numElements + threadsPerBlock - 1) / (threadsPerBlock);
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    r2c_naive<<<blocksPerGrid, threadsPerBlock>>>((struct unit *) d_A, d_B, numElements);
    cudaDeviceSynchronize();
    gettimeofday(&t0, NULL); 
    r2c_naive<<<blocksPerGrid, threadsPerBlock>>>((struct unit *) d_A, d_B, numElements);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    r2c_mod<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B2, numElements);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    
    err = cudaGetLastError();
    long elapsed = (t1.tv_sec-t0.tv_sec)*1000000 + t1.tv_usec-t0.tv_usec;
    long elapsed2 = (t2.tv_sec-t1.tv_sec)*1000000 + t2.tv_usec-t1.tv_usec;
    printf("execution time 1: %ld us\n", elapsed);
    printf("execution time 2: %ld us\n", elapsed2);

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

/*
    printf("h_B:\n");
    for (int i = 0; i < numElements; ++i)
      printf("%d ", h_B[i]);
    printf("\n");

    printf("h_B2:\n");
    for (int i = 0; i < numElements; ++i)
      printf("%d ", h_B2[i]);
    printf("\n");
*/

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

