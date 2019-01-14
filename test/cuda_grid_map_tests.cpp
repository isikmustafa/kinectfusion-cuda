#include "pch.h"

class CudaGridMapTests : public ::testing::Test
{
protected:
    const unsigned int frame_width = 640;
    const unsigned int frame_height = 480;
    const unsigned int frame_width_small = 4;
    const unsigned int frame_height_small = 4;
    const unsigned int n_pyramid_layers = 3;
    cudaChannelFormatDesc vector_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc depth_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc raw_description = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
};

TEST_F(CudaGridMapTests, TestCreate3LayerPyramid)
{
    GridMapPyramid<CudaGridMap> pyramid1(frame_width_small, frame_height_small, n_pyramid_layers, vector_description);
    GridMapPyramid<CudaGridMap> pyramid2(frame_width_small, frame_height_small, n_pyramid_layers, vector_description);

    ASSERT_NE(pyramid1[0].getCudaSurfaceObject(), pyramid1[1].getCudaSurfaceObject());
    ASSERT_NE(pyramid1[0].getCudaSurfaceObject(), pyramid2[0].getCudaSurfaceObject());
}

// For this test, place some depth image called frame.png in the "project" directory
TEST_F(CudaGridMapTests, TestUpdateDepthMapFromFile)
{
    DepthMap depth_map(frame_width, frame_height, raw_description);
    const std::string depth_frame_path = "../../frame.png";
    depth_map.update(depth_frame_path);
}

TEST_F(CudaGridMapTests, TestUpdateDepthMap)
{
    float source_data[16] = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0 };
    DepthMap depth_map(frame_width_small, frame_height_small , depth_description);
    depth_map.update((void*) source_data);
    
    float dest_data[16] = { 0.0 };
    HANDLE_ERROR(cudaMemcpyFromArray(dest_data, depth_map.getCudaArray(), 0, 0, 16 * 4, cudaMemcpyDeviceToHost));

    for (int i = 0; i < 16; i++)
    {
        ASSERT_FLOAT_EQ(source_data[i], dest_data[i]);
    }
}