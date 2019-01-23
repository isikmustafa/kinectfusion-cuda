#pragma once
#include <cusolverDn.h>

class LinearLeastSquares
{
public:
    LinearLeastSquares();
    ~LinearLeastSquares();

    /*
        Solves a linear least squares problem of the form: A * x = b,
        where A is a n_equations x 6 matrix, x is a vector of length x and b is a vector of length n_equations.
        All three arrays (mat_a_tranp, vec_b and result_x) have to be located on device memory.
        The function expects the transpose of A in column major format.
    */
    void solve(float *mat_a_transp, float *vec_b, unsigned int n_equations, float *result_x);
    int getErrorInfo();

private:
    const unsigned int m_n_variables = 6;
    cusolverDnHandle_t m_handle;
    int m_workspace_size;
    cublasFillMode_t m_fillmode = CUBLAS_FILL_MODE_LOWER;

    // Device buffers for the cuSolver info flag, workspace and the squared coefficient matrix are allocated once
    int *m_info;
    float *m_workspace;
    float *m_coef_mat;
};

