#include "pch.h"

#include "ndt_map_3d.h"
#include "ndt_pose_estimator.h"
#include "data_helper.h"

class NdtTests : public ::testing::Test
{

};

TEST_F(NdtTests, TestNdtMapUpdate)
{
    std::array<glm::fvec3, 4> points{ { { 1.0f, 2.0f, 3.0f },
                                        { 2.0f, 1.0f, 3.0f },
                                        { 2.0f, 3.0f, 1.0f },
                                        { 3.0f, 2.0f, 1.0f } } };
    float voxel_size = 5.0f;
    Coords3D coords{ 1, 1, 1 };
    
    glm::fvec3 true_mean(-0.5f, -0.5f, -0.5f);
    glm::fvec3 true_co_moments_diag(0.0);
    glm::fvec3 true_co_moments_triangle(0.0);
    for (const auto point : points)
    {
        glm::fvec3 point_local = point - glm::vec3(2.5f, 2.5f, 2.5f);
        true_co_moments_diag += (point_local - true_mean) * (point_local - true_mean);
        true_co_moments_triangle.x += (point_local.x - true_mean.x) * (point_local.y - true_mean.y);
        true_co_moments_triangle.y += (point_local.x - true_mean.x) * (point_local.z - true_mean.z);
        true_co_moments_triangle.z += (point_local.y - true_mean.y) * (point_local.z - true_mean.z);
    }

    NdtMap3D<2> ndt_map(voxel_size);
    for (glm::fvec3 point : points)
    {
        ndt_map.updateMap(point);
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

TEST_F(NdtTests, TestInit)
{
    glm::fvec4 data[6] = { 
        glm::fvec4(1.0f), 
        glm::fvec4(2.0f), 
        glm::fvec4(3.0f), 
        glm::fvec4(4.0f), 
        glm::fvec4(5.0f), 
        glm::fvec4(6.0f) };

    NdtPoseEstimator<2> estimator(2, 3, 1.0);
    estimator.initialize(data);
}