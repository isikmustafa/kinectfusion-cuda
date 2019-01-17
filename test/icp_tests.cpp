#include "pch.h"

class IcpTests : public ::testing::Test
{
protected:
    const double pi = 3.14159265358979323846;
    int width = 2;
    int height = 2;
    int n_iterations = 2;
    cudaChannelFormatDesc format_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    std::vector<unsigned int> iters_per_layer = { 1, 2, 3 };
    std::vector<void *> cuda_pointers_to_free;
    
    // 90 degree turn to the right
    glm::mat3x3 rotation_mat = glm::mat3x3(
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(-1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));

    virtual void TearDown()
    {
        for (const auto ptr : cuda_pointers_to_free)
        {
            cudaFree(ptr);
        }
    }
};

TEST_F(IcpTests, TestInitialization)
{
    ICP icp(iters_per_layer, 1.0, 1.0);
}

TEST_F(IcpTests, TestComputeCorrespondence)
{
    std::array<std::array<float, 4>, 4> vertices = { { { 1.0,  1.0, 3.0, 0.0 },
                                                       { 2.0,  2.0, 6.0, 0.0 },
                                                       {-1.0,  1.0, 3.0, 0.0 },
                                                       {-3.0, -3.0, 4.0, 0.0 } } };
    glm::mat3x3 sensor_intrinsics(
        glm::vec3(2.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 1.0f));
    
    glm::vec3 translation_vec(0.0);
    
    std::array<std::array<int, 2>, 4> true_pixel_coordinates = { { { 0,  1 },
                                                                   { 0,  1 },
                                                                   { 0,  0 },
                                                                   { 2, -1 } } };

    CudaGridMap vertex_map(width, height, format_description);
    int n_bytes = width * height * 16;
    HANDLE_ERROR(cudaMemcpyToArray(vertex_map.getCudaArray(), 0, 0, &vertices, n_bytes, cudaMemcpyHostToDevice));
    
    std::array<int, 2> *result;
    HANDLE_ERROR(cudaMalloc(&result, 4 * 8));
    cuda_pointers_to_free.push_back(result);
    
    computeCorrespondenceTestWrapper(result, vertex_map, rotation_mat, translation_vec, sensor_intrinsics);
    
    std::array<std::array<int, 2>, 4> coordinates;
    HANDLE_ERROR(cudaMemcpy(&coordinates, result, 4 * 8, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(true_pixel_coordinates[i][0], coordinates[i][0]);
        ASSERT_EQ(true_pixel_coordinates[i][1], coordinates[i][1]);
    }
}

TEST_F(IcpTests, TestNormalsAreTooDifferent)
{
    // Vectors in x-y-plane
    glm::vec3 target_normal(1.0, 0.0, 0.0);

    // The following vectors will be turned by 90 degrees to the right
    glm::vec3 normal_close_enough(0.0, 1.0, 0.0);
    glm::vec3 normal_too_different(0.0, -1.0, 0.0);
    
    float angle_thresh = pi / 2;

    bool close = normalsAreTooDifferentTestWrapper(normal_close_enough, target_normal, rotation_mat, angle_thresh);
    bool far_off = normalsAreTooDifferentTestWrapper(normal_too_different, target_normal, rotation_mat, angle_thresh);

    ASSERT_FALSE(close);
    ASSERT_TRUE(far_off);
}

TEST_F(IcpTests, TestComputeAndFillA)
{
    glm::vec3 vertex(2.0f, 3.0f, 1.0f);
    glm::vec3 normal(1.0f, 4.0f, 2.0f);

    std::array<float, 6> true_mat_a = { -2, 3, -5, 1, 4, 2 };

    std::array<float, 6> mat_a;
    computeAndFillATestWrapper(&mat_a, vertex, normal);

    for (int i = 0; i < 6; i++)
    {
        ASSERT_FLOAT_EQ(true_mat_a[i], mat_a[i]);
    }
}

TEST_F(IcpTests, TestComputeAndFillB)
{
    glm::vec3 vertex(2.0f, 3.0f, 1.0f);
    glm::vec3 target_normal(1.0f, 4.0f, 2.0f);
    glm::vec3 target_vertex(1.0f, 2.0f, 0.0f);

    float true_b = -7.0;

    float b = computeAndFillBTestWrapper(vertex, target_vertex, target_normal);

    ASSERT_FLOAT_EQ(true_b, b);
}