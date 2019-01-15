#include "pch.h"

class IcpTests : public ::testing::Test
{
protected:
    int width = 4;
    int height = 4;
    int n_iterations = 2;
    float distance_thresh = 1.0;
    float angle_thresh = 1.0;
    cudaChannelFormatDesc format_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    std::vector<unsigned int> iters_per_layer = { 1, 2, 3 };
};

TEST_F(IcpTests, TestInitialization)
{
    ICP icp(iters_per_layer, distance_thresh, angle_thresh);
}