#include "pch.h"
#include <cuda_runtime.h>
#include <cublas.h>

#include "rigid_transform_3d.h"
#include "cuda_grid_map.h"
#include "cuda_utils.h"
#include "cuda_wrapper.cuh"
#include "general_helper.h"

class TransformationTests : public ::testing::Test
{
protected:
    const double pi = 3.14159265358979323846;
    glm::mat3 true_rot_mat = glm::mat3(
        glm::vec3(0.0, 1.0, 0.0),
        glm::vec3(-1.0, 0.0, 0.0),
        glm::vec3(0.0, 0.0, 1.0));
    glm::vec3 true_trans_vec = glm::vec3(1.0, 2.0, 3.0);
    glm::mat4 true_full_transform = glm::mat4(
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