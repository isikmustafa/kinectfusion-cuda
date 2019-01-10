#include "pch.h"

#include "cuda_grid_map.cpp"

class VectorMapTests : public ::testing::Test
{
protected:
    const unsigned int frame_width = 640;
    const unsigned int frame_height = 480;
    cudaChannelFormatDesc channel_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

};

TEST_F(VectorMapTests, TestInstanziation)
{
    CudaGridMap vector_map(frame_width, frame_height, channel_description);
}