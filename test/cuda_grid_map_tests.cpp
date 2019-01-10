#include "pch.h"

#include "cuda_grid_map.cpp"

class CudaGridMapTests : public ::testing::Test
{
protected:
    const unsigned int frame_width = 64;
    const unsigned int frame_height = 48;
    cudaChannelFormatDesc channel_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    std::array<CudaGridMap*, 3> pyramid1;
    std::array<CudaGridMap*, 3> pyramid2;

    virtual void TearDown() 
    {
        for (int i = 0; i < 3; i++)
        {
            delete pyramid1[i];
            delete pyramid2[i];
        }
    }
};

TEST_F(CudaGridMapTests, TestCreate3LayerPyramid)
{
    pyramid1 = CudaGridMap::create3LayerPyramid(frame_width, frame_height, channel_description);
    pyramid2 = CudaGridMap::create3LayerPyramid(frame_width, frame_height, channel_description);

    ASSERT_NE(pyramid1[0]->getCudaSurfaceObject(), pyramid1[1]->getCudaSurfaceObject());
    ASSERT_NE(pyramid1[0]->getCudaSurfaceObject(), pyramid2[0]->getCudaSurfaceObject());
}