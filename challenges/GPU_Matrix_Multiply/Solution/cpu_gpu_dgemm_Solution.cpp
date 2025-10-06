/*------------------------------------------------------------------------------------------------
This program will fill 2 NxN matrices with random numbers, compute a matrix multiply on the CPU 
and then on the GPU, compare the values for correctness, and print _SUCCESS_ (if successful).
Written by Tom Papatheodore, edited for Anvil by Anthony Ramirez
------------------------------------------------------------------------------------------------*/

#include <stdio.h>
#include <cblas.h>
#include <cublas.h>

// Macro for checking errors in CUDA API calls
#define gpuErrorCheck(call)                                                              \
do{                                                                                       \
    cudaError_t gpuErr = call;                                                             \
    if(cudaSuccess != gpuErr){                                                             \
        printf("CUDA Error - %s:%d: '%s'\n", __FILE__, __LINE__, cudaGetErrorString(gpuErr));\
        exit(0);                                                                            \
    }                                                                                     \
}while(0)

#define N 512

int main(int argc, char *argv[])
{

    // Set device to GPU 0
    gpuErrorCheck( cudaSetDevice(0) );

    /* Allocate memory for A, B, C on CPU ----------------------------------------------*/
    double *A = (double*)malloc(N*N*sizeof(double));
    double *B = (double*)malloc(N*N*sizeof(double));
    double *C = (double*)malloc(N*N*sizeof(double));

    /* Set Values for A, B, C on CPU ---------------------------------------------------*/

    // Max size of random double
    double max_value = 10.0;

    // Set A, B, C
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            A[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
            B[i*N + j] = (double)rand()/(double)(RAND_MAX/max_value);
            C[i*N + j] = 0.0;
        }
    }

    /* Allocate memory for d_A, d_B, d_C on GPU ----------------------------------------*/
    double *d_A, *d_B, *d_C;
    gpuErrorCheck( cudaMalloc(&d_A, N*N*sizeof(double)) );
    gpuErrorCheck( cudaMalloc(&d_B, N*N*sizeof(double)) );
    gpuErrorCheck( cudaMalloc(&d_C, N*N*sizeof(double)) );

    /* Copy host arrays (A,B,C) to device arrays (d_A,d_B,d_C) -------------------------*/
    gpuErrorCheck( cudaMemcpy(d_A, A, N*N*sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(d_B, B, N*N*sizeof(double), cudaMemcpyHostToDevice) );
    gpuErrorCheck( cudaMemcpy(d_C, C, N*N*sizeof(double), cudaMemcpyHostToDevice) );

    /* Perform Matrix Multiply on CPU --------------------------------------------------*/

    const double alpha = 1.0;
    const double beta = 0.0;

    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, N, N, N, alpha, A, N, B, N, beta, C, N);

    /* Perform Matrix Multiply on GPU --------------------------------------------------*/

    cublasHandle_t handle;
    cublasCreate(&handle);

    /************************************************************/
	/* TODO: Look up the cublasDgemm routine and add it here to */
	/*       perform a matrix multiply on the GPU               */
	/*                                                          */
	/* NOTE: This will be similar to the CPU dgemm above but    */
	/*       will use d_A, d_B, and d_C instead                 */ 
	/************************************************************/

    cublasStatus_t status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, d_A, N, d_B, N, &beta, d_C, N);
    if (status != CUBLAS_STATUS_SUCCESS){
        printf("cublasDgemm failed with code %d\n", status);
        return EXIT_FAILURE;
    }

	/* Copy values of d_C back from GPU and compare with values calculated on CPU ------*/

    // Copy values of d_C (computed on GPU) into host array C_fromGPU	
    double *C_fromGPU = (double*)malloc(N*N*sizeof(double));	
    gpuErrorCheck( cudaMemcpy(C_fromGPU, d_C, N*N*sizeof(double), cudaMemcpyDeviceToHost) );

    // Check if CPU and GPU give same results
    double tolerance = 1.0e-13;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            if(fabs((C[i*N + j] - C_fromGPU[i*N + j])/C[i*N + j]) > tolerance){
                printf("Element C[%d][%d] (%f) and C_fromGPU[%d][%d] (%f) do not match!\n", i, j, C[i*N + j], i, j, C_fromGPU[i*N + j]);
                return EXIT_FAILURE;
            }
        }
    }

    /* Clean up and output --------------------------------------------------------------*/

    cublasDestroy(handle);

    // Free GPU memory
    gpuErrorCheck( cudaFree(d_A) );
    gpuErrorCheck( cudaFree(d_B) );
    gpuErrorCheck( cudaFree(d_C) );

    // Free CPU memory
    free(A);
    free(B);
    free(C);
    free(C_fromGPU);

    printf("__SUCCESS__\n");

    return 0;
}
