#pragma once

#include <array>
#include <Eigen/Dense>
#include <cublas.h>

class LinearLeastSquares
{
public:
    LinearLeastSquares();
    ~LinearLeastSquares();

	/*
		Solves a linear least squares problem of the form: A * x = b,
		where A is a n_equations x 6 matrix, b is a vector of length n_equations and result is a vector of length 6.
		mat_a_transp and vec_b must be located on device memory.
		The function expects the transpose of A in column major format (which is same as A in row major).
	*/
	std::array<float, 6> solve(float* mat_a_transpose_device, float* vec_b_device, unsigned int n_equations);

	// Computes the error between m_ATb_host and m_ATA_host * m_result
	float computeError();

private:
	cublasHandle_t m_cublas_handle;

	// Two sides of normal equation and result on host.
	Eigen::Matrix<float, 6, 6> m_ATA_host;
	Eigen::Matrix<float, 6, 1> m_ATb_host;
	Eigen::Matrix<float, 6, 1> m_result;

	// Two sides of normal equation on device.
    float* m_ATA_device;
	float* m_ATb_device;
};

