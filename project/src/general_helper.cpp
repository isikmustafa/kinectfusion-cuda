#include "general_helper.h"

#undef STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "cuda_utils.h"

void cudaMatrixMatrixMultiplication(float *mat_left_transp, float *mat_right_transp, float *mat_out_transp,
    int n_rows_left, int n_cols_left, int n_cols_right, cublasOperation_t operation_right)
{
    cublasHandle_t cublas_handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    float alpha = 1.0f;
    float beta = 0.0f;

    HANDLE_CUBLAS_ERROR(cublasSgemm(cublas_handle, CUBLAS_OP_N, operation_right, n_rows_left, n_cols_right, n_cols_left,
        &alpha, mat_left_transp, n_rows_left, mat_right_transp, n_cols_right, &beta, mat_out_transp, n_rows_left));
}

void cudaMatrixVectorMultiplication(float *mat_left_transp, float *vec_right, float *vec_out, int n_rows, int n_cols)
{
    cublasHandle_t cublas_handle;
    HANDLE_CUBLAS_ERROR(cublasCreate(&cublas_handle));

    float alpha = 1.0f;
    float beta = 1.0f;

    HANDLE_CUBLAS_ERROR(cublasSgemv(cublas_handle, CUBLAS_OP_N, n_rows, n_cols, &alpha, mat_left_transp, n_rows, 
        vec_right, 1, &beta, vec_out, 1));
}

void writeSurface1x32(std::string file_name, cudaArray* gpu_source, int width, int height)
{
    std::unique_ptr<float[]> float_data(new float[width * height]);
    auto float_data_ptr = float_data.get();

    HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4, cudaMemcpyDeviceToHost));

    std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height]);
    auto byte_data_ptr = byte_data.get();

    int size = width * height;
    for (int i = 0; i < size; ++i)
    {
        byte_data_ptr[i] = static_cast<unsigned char>(float_data_ptr[i] / 200);
    }

    auto final_path = std::to_string(width) + "x" + std::to_string(height) + ".png";
    stbi_write_png(final_path.c_str(), width, height, 1, byte_data_ptr, width);
}

void writeSurface4x32(std::string file_name, cudaArray* gpu_source, int width, int height)
{
    std::unique_ptr<float[]> float_data(new float[width * height * 4]);
    auto float_data_ptr = float_data.get();

    HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4 * 4, cudaMemcpyDeviceToHost));

    std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height * 3]);
    auto byte_data_ptr = byte_data.get();

    int size = width * height;
    for (int i = 0; i < size; ++i)
    {
        int idx_byte = i * 3;
        int idx_float = i * 4;
        byte_data_ptr[idx_byte] = static_cast<unsigned char>(float_data_ptr[idx_float] * 255.0f);
        byte_data_ptr[idx_byte + 1] = static_cast<unsigned char>(float_data_ptr[idx_float + 1] * 255.0f);
        byte_data_ptr[idx_byte + 2] = static_cast<unsigned char>(float_data_ptr[idx_float + 2] * 255.0f);
    }

    auto final_path = std::to_string(width) + "x" + std::to_string(height) + file_name;
    stbi_write_png(final_path.c_str(), width, height, 3, byte_data_ptr, width * 3);
}

void writeDepthPyramidToFile(std::string file_name, std::array<CudaGridMap, 3> pyramid)
{
    for (const CudaGridMap &map : pyramid)
    {
        auto dims = map.getGridDims();
        writeSurface1x32(file_name, map.getCudaArray(), dims[0], dims[1]);
    }
}

void writeVectorPyramidToFile(std::string file_name, std::array<CudaGridMap, 3> pyramid)
{
    for (const CudaGridMap &map : pyramid)
    {
        auto dims = map.getGridDims();
        writeSurface4x32(file_name, map.getCudaArray(), dims[0], dims[1]);
    }
}