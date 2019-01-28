#include "pch.h"

#include "ndt_map_3d.h"

class NdtTests : public ::testing::Test
{

};

TEST_F(NdtTests, TestNdtMapUpdate)
{
    std::array<glm::fvec3, 4> points{ { { 1.0f, 2.0f, 3.0f },
                                        { 2.0f, 1.0f, 3.0f },
                                        { 3.0f, 2.0f, 1.0f },
                                        { 2.0f, 3.0f, 1.0f } } };
    Coords3D coords{ 1, 0, 1 };
    
    glm::fvec3 true_mean(2.0f, 2.0f, 2.0f);
    glm::fvec3 true_co_moments_diag(0.0);
    glm::fvec3 true_co_moments_triangle(0.0);
    for (const auto point : points)
    {
        true_co_moments_diag += (point - true_mean) * (point - true_mean);
        true_co_moments_triangle.x += (point.x - true_mean.x) * (point.y - true_mean.y);
        true_co_moments_triangle.y += (point.x - true_mean.x) * (point.z - true_mean.z);
        true_co_moments_triangle.z += (point.y - true_mean.y) * (point.z - true_mean.z);
    }

    NdtMap3d<2> ndt_map;
    for (const auto point : points)
    {
        ndt_map.updateVoxel(point, coords);
    }

    NdtVoxel voxel = ndt_map.getVoxel(coords);
    ASSERT_FLOAT_EQ(4.0, voxel.count);
    for (int i = 0; i < 3; i++)
    {
        ASSERT_NEAR(true_mean[i], voxel.mean[i], 1e-6f);
        ASSERT_NEAR(true_co_moments_diag[i], voxel.co_moments_diag[i], 1e-6f);
        ASSERT_NEAR(true_co_moments_triangle[i], voxel.co_moments_triangle[i], 1e-6f);
    }
}