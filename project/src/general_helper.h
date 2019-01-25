#pragma once
#include <string>
#include <array>
#include <cusolverDn.h>

#include "cuda_grid_map.h"
#include "grid_map_pyramid.h"

/*
    Matrix-matrix multiplication on device memory:
        mat_left_transp             * operation_lef(mat_right_transp) = mat_out_transp
        [n_rows_left x n_cols_left] * [n_cols_left x n_cols_right]    = [n_rows_left x n_cols_right]
*/
void cudaMatrixMatrixMultiplication(float *mat_left_transp, float *mat_right_transp, float *mat_out_transp, 
    int n_rows_left, int n_cols_left, int n_cols_right, cublasOperation_t operation_right, cublasHandle_t cublas_handle);

/*
    Matrix-vector multiplication on device memory:
        mat_left_transp   * vec_right = vec_out
        [n_rows x n_cols] * [n_cols]  = [n_rows]
*/
void cudaMatrixVectorMultiplication(float *mat_left_transp, float *vec_right, float *vec_out, int n_rows, int n_cols,
    cublasHandle_t cublas_handle);

std::pair<float,float> poseError(glm::mat4x4 pose_1, glm::mat4x4 pose_2);


/////////////// DEBUG FUNCTIONS /////////////////////////
void writeSurface1x32(std::string file_name, cudaArray* gpu_source, int width, int height);
void writeSurface4x32(std::string file_name, cudaArray* gpu_source, int width, int height);
void writeDepthPyramidToFile(std::string file_name, GridMapPyramid<CudaGridMap> pyramid);
void writeVectorPyramidToFile(std::string file_name, std::array<CudaGridMap, 3> pyramid);
