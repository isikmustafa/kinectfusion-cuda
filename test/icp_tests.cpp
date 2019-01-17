#include "pch.h"

class IcpTests : public ::testing::Test
{
protected:
    int width = 2;
    int height = 2;
    int n_iterations = 2;
    float distance_thresh = 1.0;
    float angle_thresh = 1.0;
    cudaChannelFormatDesc format_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    std::vector<unsigned int> iters_per_layer = { 1, 2, 3 };
    std::vector<void *> cuda_pointers_to_free;

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
    ICP icp(iters_per_layer, distance_thresh, angle_thresh);
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

    std::array<std::array<int, 2>, 4> true_pixel_coordinates = { { { 1, 1 },
                                                                   { 1, 1 },
                                                                   { 0, 1 },
                                                                   { -1, -1 } } };

    CudaGridMap vertex_map(width, height, format_description);
    int n_bytes = width * height * 16;
    HANDLE_ERROR(cudaMemcpyToArray(vertex_map.getCudaArray(), 0, 0, &vertices, n_bytes, cudaMemcpyHostToDevice));
    
    std::array<int, 2> *result;
    HANDLE_ERROR(cudaMalloc(&result, 4 * 8));
    cuda_pointers_to_free.push_back(result);
    
    computeCorrespondenceTestWrapper(result, vertex_map, sensor_intrinsics);
    
    std::array<std::array<int, 2>, 4> coordinates;
    HANDLE_ERROR(cudaMemcpy(&coordinates, result, 4 * 8, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 4; i++)
    {
        ASSERT_EQ(true_pixel_coordinates[i][0], coordinates[i][0]);
        ASSERT_EQ(true_pixel_coordinates[i][1], coordinates[i][1]);
    }
}