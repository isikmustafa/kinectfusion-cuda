#include "pch.h"
#include <cuda_runtime.h>
#include "rigid_transform_3d.h"
#include "cuda_grid_map.h"
#include "cuda_utils.h"
#include "cuda_wrapper.cuh"
#include "general_helper.h"

class TransformationTests : public ::testing::Test
{
protected:
    const double pi = 3.14159265358979323846;
    glm::mat3x3 true_rot_mat = glm::mat3x3(
        glm::vec3(0.0, 1.0, 0.0),
        glm::vec3(-1.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, 1.0));
    glm::vec3 true_trans_vec = glm::vec3(1.0, 2.0, 3.0);
    glm::mat4x4 true_full_transform = glm::mat4x4(
        glm::vec4(0.0, 1.0, 0.0, 0.0),
        glm::vec4(-1.0, 0.0, 0.0, 0.0),
        glm::vec4(0.0, 0.0, 1.0, 0.0),
        glm::vec4(1.0, 2.0, 3.0, 1.0));

    unsigned int width = 2;
    unsigned int height = 2;
    cudaChannelFormatDesc format_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    std::vector<void *> cuda_pointers_to_free;

    virtual void TearDown()
    {
        for (const auto ptr : cuda_pointers_to_free)
        {
            cudaFree(ptr);
        }
    }
};

TEST_F(TransformationTests, TestComputeNormal)
{
    std::array<std::array<float, 4>, 4> vertices = { { { 0.0,  0.0, 0.0, 0.0 },
                                                       { 2.0,  0.0, 0.0, 0.0 },
                                                       { 0.0,  4.0, 0.0, 0.0 },
                                                       { 3.0,  3.0, 3.0, 0.0 } } };
    glm::vec3 true_normal(0.0f, 0.0f, 1.0f);

    CudaGridMap vertex_map(width, height, format_description);
    int n_bytes = width * height * 16;
    HANDLE_ERROR(cudaMemcpyToArray(vertex_map.getCudaArray(), 0, 0, &vertices, n_bytes, cudaMemcpyHostToDevice));

    glm::vec3 result_normal = computeNormalTestWrapper(vertex_map, 0, 0);

    for (int i = 0; i < 3; i++)
    {
        ASSERT_EQ(true_normal[0], result_normal[0]);
    }
}

TEST_F(TransformationTests, TestMatrixMatrixMultiply)
{
    std::array<std::array<float, 6>, 2> mat_a = { { { 1.0, 2.0, 0.0, 1.0, 0.0, 2.0 },
                                                    { 0.0, 1.0, 2.0, 1.0, 0.0, 0.0 } } };

    std::array<std::array<float, 6>, 2> *mat_a_device;
    HANDLE_ERROR(cudaMalloc(&mat_a_device, sizeof(std::array<std::array<float, 6>, 2>)));
    cuda_pointers_to_free.push_back(mat_a_device);
    HANDLE_ERROR(cudaMemcpy(mat_a_device, &mat_a, sizeof(std::array<std::array<float, 6>, 2>), cudaMemcpyHostToDevice));

    std::array<std::array<float, 6>, 6> true_result = { { { 1.0, 2.0, 0.0, 1.0, 0.0, 2.0 },
                                                          { 2.0, 5.0, 2.0, 3.0, 0.0, 4.0 },
                                                          { 0.0, 2.0, 4.0, 2.0, 0.0, 0.0 },
                                                          { 1.0, 3.0, 2.0, 2.0, 0.0, 2.0 },
                                                          { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                                                          { 2.0, 4.0, 0.0, 2.0, 0.0, 4.0 } } };

    std::array<std::array<float, 6>, 6> *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(std::array<std::array<float, 6>, 6>)));
    cuda_pointers_to_free.push_back(result_device);

    cudaMatrixMatrixMultiplication((float *)mat_a_device, (float *)mat_a_device, (float *)result_device, 6, 2, 6, 
        CUBLAS_OP_T);

    std::array<std::array<float, 6>, 6> result_host;
    HANDLE_ERROR(cudaMemcpy(&result_host, result_device, sizeof(std::array<std::array<float, 6>, 6>),
        cudaMemcpyDeviceToHost));

    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            ASSERT_FLOAT_EQ(true_result[i][j], result_host[i][j]);
        }
    }
}

TEST_F(TransformationTests, TestMatrixVectorMultiply)
{
    std::array<std::array<float, 6>, 2> mat_a = { { { 1.0, 2.0, 0.0, 1.0, 0.0, 2.0 },
                                                    { 0.0, 1.0, 2.0, 1.0, 0.0, 0.0 } } };

    std::array<std::array<float, 6>, 2> *mat_a_device;
    HANDLE_ERROR(cudaMalloc(&mat_a_device, sizeof(std::array<std::array<float, 6>, 2>)));
    cuda_pointers_to_free.push_back(mat_a_device);
    HANDLE_ERROR(cudaMemcpy(mat_a_device, &mat_a, sizeof(std::array<std::array<float, 6>, 2>), cudaMemcpyHostToDevice));

    std::array<float, 2> vec_b = { 1, 1 };
    std::array<float, 2> *vec_b_device;
    HANDLE_ERROR(cudaMalloc(&vec_b_device, sizeof(std::array<float, 2>)));
    cuda_pointers_to_free.push_back(vec_b_device);
    HANDLE_ERROR(cudaMemcpy(vec_b_device, &vec_b, sizeof(std::array<float, 2>), cudaMemcpyHostToDevice));

    std::array<float, 6> true_result = { 1.0, 3.0, 2.0, 2.0, 0.0, 2.0 };


    std::array<float, 6> *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(std::array<float, 6>)));
    cuda_pointers_to_free.push_back(result_device);

    cudaMatrixVectorMultiplication((float *)mat_a_device, (float *)vec_b_device, (float *)result_device, 6, 2);

    std::array<float, 6> result_host;
    HANDLE_ERROR(cudaMemcpy(&result_host, result_device, sizeof(std::array<float, 6>), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 6; i++)
    {
        ASSERT_FLOAT_EQ(true_result[i], result_host[i]);
    }
}