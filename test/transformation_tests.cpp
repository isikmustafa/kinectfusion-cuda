#include "pch.h"

#include <cuda_runtime.h>
#include "glm_macro.h"
#include "glm/glm.hpp"

#include "rigid_transform_3d.cpp"

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

TEST_F(TransformationTests, TestGetSetFull)
{
    RigidTransform3D transform;

    transform.setHomoMat(true_full_transform);
    glm::mat4x4 *homo_mat = transform.getHomoMat();
    
    cudaPointerAttributes pointer_attributes;
    cudaPointerGetAttributes(&pointer_attributes, homo_mat);
    ASSERT_EQ(cudaMemoryTypeDevice, pointer_attributes.type);

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