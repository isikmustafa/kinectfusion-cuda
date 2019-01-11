#include "pch.h"

#include <cuda_runtime.h>
#include "glm_macro.h"
#include "glm/glm.hpp"

#include "rigid_transform_3d.cpp"
#include "cuda_utils.h"

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
};

TEST_F(TransformationTests, TestSetFromHost)
{
    RigidTransform3D transform;

    transform.setHomoMat(&true_full_transform);
    glm::mat4x4 *homo_mat = transform.getHomoMat();
    
    ASSERT_TRUE(isDeviceMemory(homo_mat));

    glm::mat4x4 *host_homo_mat = (glm::mat4x4*)malloc(sizeof(glm::mat4x4));
    cudaMemcpy(host_homo_mat, homo_mat, sizeof(glm::mat4x4), cudaMemcpyDeviceToHost);

    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            ASSERT_FLOAT_EQ((*host_homo_mat)[col][row], true_full_transform[col][row]);
        }
    }
}

TEST_F(TransformationTests, TestSetFromDevice)
{
    glm::mat4x4 *device_true_mat;
    cudaMalloc((void**) &device_true_mat, sizeof(glm::mat4x4));
    cudaMemcpy(device_true_mat, &true_full_transform, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
    
    RigidTransform3D transform;

    transform.setHomoMat(device_true_mat);
    glm::mat4x4 *homo_mat = transform.getHomoMat();

    ASSERT_TRUE(isDeviceMemory(homo_mat));

    glm::mat4x4 *host_homo_mat = (glm::mat4x4*)malloc(sizeof(glm::mat4x4));
    cudaMemcpy(host_homo_mat, homo_mat, sizeof(glm::mat4x4), cudaMemcpyDeviceToHost);

    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            ASSERT_FLOAT_EQ((*host_homo_mat)[col][row], true_full_transform[col][row]);
        }
    }
}