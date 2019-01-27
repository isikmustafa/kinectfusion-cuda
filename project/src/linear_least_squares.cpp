#include "linear_least_squares.h"

#include <cuda_runtime.h>

#include "cuda_utils.h"
#include "general_helper.h"
#include "timer.h"

LinearLeastSquares::LinearLeastSquares()
{
    HANDLE_CUBLAS_ERROR(cublasCreate(&m_cublas_handle));
    HANDLE_CUSOLVER_ERROR(cusolverDnCreate(&m_cusolver_handle));

    HANDLE_ERROR(cudaMalloc(&m_coef_mat, m_n_variables *m_n_variables * sizeof(float)));
    HANDLE_ERROR(cudaMalloc((void **)& m_info, sizeof(int)));

    HANDLE_CUSOLVER_ERROR(cusolverDnSpotrf_bufferSize(m_cusolver_handle, m_fillmode, m_n_variables, nullptr, 
        m_n_variables, &m_workspace_size));
    HANDLE_ERROR(cudaMalloc((void **)&m_workspace, m_workspace_size * sizeof(float)));
}

LinearLeastSquares::~LinearLeastSquares()
{
    HANDLE_ERROR(cudaFree(m_coef_mat));
    HANDLE_ERROR(cudaFree(m_workspace));
    HANDLE_ERROR(cudaFree(m_info));
}

float LinearLeastSquares::solve(float *mat_a_transp, float *vec_b, unsigned int n_equations, float *result_x)
{
    Timer timer;
    timer.start();

    // Use the result vector as buffer for the result
    float *bias_vec = result_x;

    // Calcuate the squared coefficient matrix and the bias vector for the system: coeff_mat * x = bias_vec
    //      coeff_mat = transpose(mat_a) * mat_a
    //      bias_vec = transpose(mat_a) * vec_b
    cudaMatrixMatrixMultiplication(mat_a_transp, mat_a_transp, m_coef_mat, m_n_variables, n_equations, 
        m_n_variables, CUBLAS_OP_T, m_cublas_handle);
    cudaMatrixVectorMultiplication(mat_a_transp, vec_b, bias_vec, m_n_variables, n_equations, m_cublas_handle);
	
    // Save copies of coeff_mat and bias_vec for the error calculation later
    std::array<std::array<float, 6>, 6> coef_mat_host;
    std::array<float, 6> bias_vec_host;
	HANDLE_ERROR(cudaMemcpy(&coef_mat_host, m_coef_mat, sizeof(std::array<std::array<float, 6>, 6>), 
        cudaMemcpyDeviceToHost));
	HANDLE_ERROR(cudaMemcpy(&bias_vec_host, bias_vec, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    // Cholesky decomposition of coeff_mat = L * L^T, lower triangle of coeff_mat is replaced by the factor L
    HANDLE_CUSOLVER_ERROR(cusolverDnSpotrf(m_cusolver_handle, m_fillmode, m_n_variables, m_coef_mat, m_n_variables, 
        m_workspace, m_workspace_size, m_info));
    
    // Solve coeff_mat * x = bias_vec , where coeff_mat is cholesky factorized, bias_vec is overwritten by the solution
    HANDLE_CUSOLVER_ERROR(cusolverDnSpotrs(m_cusolver_handle, m_fillmode, m_n_variables, 1, m_coef_mat, m_n_variables, 
        bias_vec, m_n_variables, m_info));

    // Copy also result to host to compute error
	std::array<float, 6> vec_x_host;
	HANDLE_ERROR(cudaMemcpy(&vec_x_host, result_x, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    computeError(coef_mat_host, vec_x_host, bias_vec_host);

    // Finished: result_x already points to the solution
    
    return timer.getTime() * 1000.0;
}

int LinearLeastSquares::getCuSolverErrorInfo()
{
    int info;
    HANDLE_ERROR(cudaMemcpy(&info, m_info, sizeof(int), cudaMemcpyDeviceToHost));
    return info;
}

float LinearLeastSquares::getLastError()
{
    return m_last_error;
}

void LinearLeastSquares::computeError(std::array<std::array<float, 6>, 6> &coef_mat, std::array<float, 6> vec_x,
    std::array<float, 6> &bias_vec)
{
    m_last_error = 0;
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            m_last_error += coef_mat[i][j] * vec_x[i] - bias_vec[i];
        }
    }
}
