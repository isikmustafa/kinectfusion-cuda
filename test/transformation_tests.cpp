#include "pch.h"

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
};

TEST_F(TransformationTests, TestSetGet)
{
    RigidTransform3D transform;

    transform.setHomoMat(true_full_transform);
    glm::mat4x4 homo_mat = transform.getHomoMat();

    for (int col = 0; col < 4; col++)
    {
        for (int row = 0; row < 4; row++)
        {
            ASSERT_FLOAT_EQ(homo_mat[col][row], true_full_transform[col][row]);
        }
    }
}

glm::vec3 foo()
{
    glm::vec3 central_vertex(1.0, 2.0, 3.0);
    glm::vec3 next_in_row(3.0, 2.0, 1.0);
    glm::vec3 next_in_column(2.0, 1.0, 3.0);
    return glm::normalize(glm::cross(next_in_row - central_vertex, next_in_column - central_vertex));
}


TEST_F(TransformationTests, TestComputeNormal)
{
    std::array<std::array<float, 4>, 4> vertices = { { { 0.0,  0.0, 0.0, 0.0 },
                                                       { 2.0,  0.0, 0.0, 0.0 },
                                                       { 0.0,  4.0, 0.0, 0.0 },
                                                       { 3.0,  3.0, 3.0, 0.0 } } };
    glm::vec3 true_normal(0.0f, 0.0f, 1.0f);
    glm::vec3 bar = foo();

    CudaGridMap vertex_map(width, height, format_description);
    int n_bytes = width * height * 16;
    HANDLE_ERROR(cudaMemcpyToArray(vertex_map.getCudaArray(), 0, 0, &vertices, n_bytes, cudaMemcpyHostToDevice));

    glm::vec3 result_normal = computeNormalTestWrapper(vertex_map, 0, 0);

    for (int i = 0; i < 3; i++)
    {
        ASSERT_EQ(true_normal[0], result_normal[0]);
    }
}