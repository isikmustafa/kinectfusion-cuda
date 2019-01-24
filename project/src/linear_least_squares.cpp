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

    // Cholesky decomposition of coeff_mat = L * L^T, lower triangle of coeff_mat is replaced by the factor L
    HANDLE_CUSOLVER_ERROR(cusolverDnSpotrf(m_cusolver_handle, m_fillmode, m_n_variables, m_coef_mat, m_n_variables, m_workspace,
        m_workspace_size, m_info));
    
    // Solve coeff_mat * x = bias_vec , where coeff_mat is cholesky factorized, bias_vec is overwritten by the solution
    HANDLE_CUSOLVER_ERROR(cusolverDnSpotrs(m_cusolver_handle, m_fillmode, m_n_variables, 1, m_coef_mat, m_n_variables, bias_vec, 
        m_n_variables, m_info));
    HANDLE_ERROR(cudaDeviceSynchronize());

    // Finished: result_x already points to the solution
    return timer.getTime() * 1000.0;
}

int LinearLeastSquares::getErrorInfo()
{
    int info;
    HANDLE_ERROR(cudaMemcpy(&info, m_info, sizeof(int), cudaMemcpyDeviceToHost));
    return info;
}
