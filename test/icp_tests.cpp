#include "pch.h"

class IcpTests : public ::testing::Test
{
protected:
    int width = 4;
    int height = 4;
    int n_iterations = 2;
    float distance_thresh = 1.0;
    float angle_thresh = 1.0;
};

TEST_F(IcpTests, TestInitialization)
{
    cudaChannelFormatDesc depth_description = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc vertex_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    GridMapPyramid<DepthMap> depth_map_pyramid(width, height, depth_description);
    GridMapPyramid<CudaGridMap> vertex_map_pyramid(width, height, vertex_description);
    RigidTransform3D previous_pose;

    ICP icp(vertex_map_pyramid, depth_map_pyramid, previous_pose, n_iterations, distance_thresh, angle_thresh);
}