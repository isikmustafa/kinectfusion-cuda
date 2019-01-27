#pragma once
#include <array>
#include <cusolverDn.h>

class LinearLeastSquares
{
public:
    LinearLeastSquares();
    ~LinearLeastSquares();

    /*
        Solves a linear least squares problem of the form: A * x = b,
        where A is a n_equations x 6 matrix, x is a vector of length 6 and b is a vector of length n_equations.
        All three arrays (mat_a_transp, vec_b and result_x) must be located on device memory.
        The function expects the transpose of A in column major format (which is same as A in row major).
    */
    float solve(float *mat_a_transp, float *vec_b, unsigned int n_equations, float *result_x);
    
    // Returns cuSolver error code of last call
    int getCuSolverErrorInfo();

    // Returns error (transpose(A) * A * x - transpose(A) *b) of the last call to solve()
    float getLastError();

private:
    const unsigned int m_n_variables = 6;
    int m_workspace_size;
    float m_last_error;
    cublasHandle_t m_cublas_handle;
    cusolverDnHandle_t m_cusolver_handle;
    cublasFillMode_t m_fillmode = CUBLAS_FILL_MODE_LOWER;

    // Device buffers for the cuSolver info flag, workspace and the squared coefficient matrix are allocated once
    int *m_info;
    float *m_workspace;
    float *m_coef_mat;

private:
    void computeError(std::array<std::array<float, 6>, 6> &coef_mat, std::array<float, 6> vec_x, 
        std::array<float, 6> &bias_vec);
};

