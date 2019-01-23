#pragma once

#include <iostream>
#include <cuda_runtime.h>
#include <cusolverDn.h>

static void handleError(cudaError_t err, const char* file, int line)
{
	if (err != cudaSuccess)
	{
		std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line << std::endl;
		system("PAUSE");
	}
}

static void handleCuBlasError(cublasStatus_t err, const char* file, int line)
{
    if (err != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS error in " << file << " at line " << line << std::endl;
        system("PAUSE");
    }
}

static void handleCuSolverError(cusolverStatus_t err, const char* file, int line)
{
    if (err != CUSOLVER_STATUS_SUCCESS)
    {
        std::cerr << "cuSolver error in " << file << " at line " << line << std::endl;
        system("PAUSE");
    }
}

static bool isDeviceMemory(void* ptr)
{
    cudaPointerAttributes pointer_attributes;
    cudaPointerGetAttributes(&pointer_attributes, ptr);

	return pointer_attributes.type == cudaMemoryTypeDevice;
}

#define HANDLE_ERROR( err ) (handleError( err, __FILE__, __LINE__ ))
#define HANDLE_CUBLAS_ERROR( err ) (handleCuBlasError( err, __FILE__, __LINE__ ))
#define HANDLE_CUSOLVER_ERROR( err ) (handleCuSolverError( err, __FILE__, __LINE__ ))