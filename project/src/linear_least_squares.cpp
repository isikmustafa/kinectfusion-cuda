#include "linear_least_squares.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "general_helper.h"
#include "timer.h"

LinearLeastSquares::LinearLeastSquares()
{
    HANDLE_CUBLAS_ERROR(cublasCreate(&m_cublas_handle));
    HANDLE_ERROR(cudaMalloc(&m_ATA_device, 36 * sizeof(float)));
	HANDLE_ERROR(cudaMalloc(&m_ATb_device, 6 * sizeof(float)));
}

LinearLeastSquares::~LinearLeastSquares()
{
	HANDLE_CUBLAS_ERROR(cublasDestroy(m_cublas_handle));
    HANDLE_ERROR(cudaFree(m_ATA_device));
	HANDLE_ERROR(cudaFree(m_ATb_device));
}

std::array<float, 6> LinearLeastSquares::solve(float* mat_a_transpose_device, float* vec_b_device, unsigned int n_equations)
{
	float alpha = 1.0f;
	float beta = 0.0f;

	// Compute m_ATA_device = transpose(A) * A  where transpose(A) = mat_a_transpose_device
	HANDLE_CUBLAS_ERROR(cublasSgemm(m_cublas_handle, CUBLAS_OP_N, CUBLAS_OP_T, 6, 6, n_equations,
		&alpha, mat_a_transpose_device, 6, mat_a_transpose_device, 6, &beta, m_ATA_device, 6));

	// Compute m_ATb_device = transpose(A) * b  where transpose(A) = mat_a_transpose_device and b = vec_b_device
	HANDLE_CUBLAS_ERROR(cublasSgemv(m_cublas_handle, CUBLAS_OP_N, 6, n_equations, &alpha, mat_a_transpose_device, 6,
		vec_b_device, 1, &beta, m_ATb_device, 1));
	
	// Copy data from device to host.
	HANDLE_ERROR(cudaMemcpy(m_ATA_host.data(), m_ATA_device, 36 * sizeof(float), cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(m_ATb_host.data(), m_ATb_device, 6 * sizeof(float), cudaMemcpyDeviceToHost));

	// Solve the normal equation on host.
	m_result = m_ATA_host.llt().solve(m_ATb_host);
    
	return { m_result(0), m_result(1), m_result(2), m_result(3), m_result(4), m_result(5) };
}

float LinearLeastSquares::computeError()
{
	return (m_ATA_host * m_result - m_ATb_host).norm();
}