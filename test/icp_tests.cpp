#include "pch.h"

#include <cuda_runtime.h>
#include <cublas.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "rigid_transform_3d.h"
#include "icp.h"
#include "cuda_wrapper.cuh"
#include "icp.cuh"
#include "cuda_utils.h"
#include "linear_least_squares.h"
#include "depth_map.h"
#include "measurement.cuh"
#include "sensor.h"
#include "general_helper.h"
#include "rgbd_dataset.h"


class IcpTests : public ::testing::Test
{
protected:
    const float pi = 3.14159265358979323846;
    int width = 2;
    int height = 2;
    int n_iterations = 2;
    cudaChannelFormatDesc format_description = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    std::vector<unsigned int> iters_per_layer = { 10, 4, 2 };
    std::vector<void *> cuda_pointers_to_free;
    std::streambuf* oldCoutStreamBuf = std::cout.rdbuf();

    glm::mat3x3 intrinsics = glm::mat3x3(
        glm::vec3(2.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 1.0f));
    
    // 90 degree turn to the right
    glm::mat3x3 rotation_mat = glm::mat3x3(
        glm::vec3(0.0f, -1.0f, 0.0f),
        glm::vec3(1.0f,  0.0f, 0.0f),
        glm::vec3(0.0f,  0.0f, 1.0f));

    virtual void TearDown()
    {
        for (const auto ptr : cuda_pointers_to_free)
        {
            cudaFree(ptr);
        }

        std::cout.rdbuf(oldCoutStreamBuf);
    }
};

TEST_F(IcpTests, TestInitialization)
{
    RigidTransform3D transform;

    ICP icp(iters_per_layer, 4, 4, 1.0, 1.0);
}

TEST_F(IcpTests, TestComputeCorrespondence)
{
    std::array<glm::vec3, 4> vertices = { { { 1.0,  1.0, 3.0 },
                                            { 2.0,  2.0, 6.0 },
                                            {-1.0,  1.0, 3.0 },
                                            {-3.0, -3.0, 4.0 } } };
    
    glm::vec3 translation_vec(0.0);
    
    std::array<std::array<int, 2>, 4> true_pixel_coordinates = { { { 0,  1 },
                                                                   { 0,  1 },
                                                                   { 0,  0 },
                                                                   { 2, -1 } } };
    
    std::array<glm::vec2, 4> coordinates;
    for (int i = 0; i < 4; i++)
    {
        coordinates[i] = computeCorrespondenceTestWrapper(vertices[i], rotation_mat, translation_vec, intrinsics);
    }

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

TEST_F(IcpTests, TestSolveLinearSystem)
{
    std::array<std::array<float, 6>, 7> mat_a = { { { 1.0, 0.0, 0.0, 0.0, 0.0, 0.0 },
                                                    { 0.0, 2.0, 0.0, 0.0, 0.0, 0.0 },
                                                    { 0.0, 0.0, 3.0, 0.0, 0.0, 0.0 },
                                                    { 0.0, 0.0, 0.0, 4.0, 0.0, 0.0 },
                                                    { 0.0, 0.0, 0.0, 0.0, 5.0, 0.0 },
                                                    { 0.0, 0.0, 0.0, 0.0, 0.0, 6.0 },
                                                    { 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 } } };
    std::array<std::array<float, 6>, 7> *mat_a_device;
    HANDLE_ERROR(cudaMalloc(&mat_a_device, sizeof(std::array<std::array<float, 6>, 7>)));
    cuda_pointers_to_free.push_back(mat_a_device);
    HANDLE_ERROR(cudaMemcpy(mat_a_device, &mat_a, sizeof(std::array<std::array<float, 6>, 7>), cudaMemcpyHostToDevice));

    std::array<float, 7> vec_b = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 6.0 };
    std::array<float, 7> *vec_b_device;
    HANDLE_ERROR(cudaMalloc(&vec_b_device, sizeof(std::array<float, 7>)));
    cuda_pointers_to_free.push_back(vec_b_device);
    HANDLE_ERROR(cudaMemcpy(vec_b_device, &vec_b, sizeof(std::array<float, 7>), cudaMemcpyHostToDevice));

    std::array<float, 6> *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(std::array<float, 6>)));
    cuda_pointers_to_free.push_back(vec_b_device);

    LinearLeastSquares solver;
    solver.solve((float *)mat_a_device, (float *)vec_b_device, 7, (float *)result_device);

    std::array<float, 6> result_host;
    HANDLE_ERROR(cudaMemcpy(&result_host, result_device, sizeof(std::array<float, 6>), cudaMemcpyDeviceToHost));

    for (int i = 0; i < 6; i ++)
    {
        ASSERT_NEAR(1.0, result_host[i], 0.0001);
    }
}

TEST_F(IcpTests, TestConstructIcpResiduals)
{
    std::array<std::array<float, 4>, 4> target_vertices = { { {-1.0, -1.0, 3.0, -1.0 },
                                                              { 1.0, -1.0, 3.0, -1.0 },
                                                              {-1.0,  1.0, 3.0, -1.0 },
                                                              { 1.0,  1.0, 3.0, -1.0 } } };
    CudaGridMap target_vertex_map(2, 2, format_description);
    int n_bytes = 16 * 2 * 2;
    HANDLE_ERROR(cudaMemcpyToArray(target_vertex_map.getCudaArray(), 0, 0, &target_vertices[0][0], n_bytes,
        cudaMemcpyHostToDevice));

    std::array<std::array<float, 4>, 4> target_normals = { { { 0.0,  0.0, -1.0, -1.0 },
                                                             { 0.0,  0.0, -1.0, -1.0 },
                                                             { 0.0,  0.0, -1.0, -1.0 },
                                                             { 0.0,  0.0, -1.0, -1.0 } } };
    CudaGridMap target_normal_map(2, 2, format_description);
    HANDLE_ERROR(cudaMemcpyToArray(target_normal_map.getCudaArray(), 0, 0, &target_normals[0][0], n_bytes,
        cudaMemcpyHostToDevice));

    std::array<std::array<float, 4>, 4> vertices = { { {-1.0, -1.0, 4.0, -1.0 },
                                                       { 1.0, -1.0, 4.0, -1.0 },
                                                       {-1.0,  1.0, 5.0, -1.0 },
                                                       { 1.0,  1.0, 4.0, -1.0 } } };
    CudaGridMap vertex_map(2, 2, format_description);
    HANDLE_ERROR(cudaMemcpyToArray(vertex_map.getCudaArray(), 0, 0, &vertices[0][0], n_bytes, cudaMemcpyHostToDevice));

    glm::mat3x3 no_rotation(
        glm::vec3(1.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        glm::vec3(0.0f, 0.0f, 1.0f));
    glm::vec3 no_translation(0.0);

    glm::mat3x3 intrinsics(
        glm::vec3(2.0f, 0.0f, 0.0f),
        glm::vec3(0.0f, 2.0f, 0.0f),
        glm::vec3(1.0f, 1.0f, 1.0f));

    float distance_threshold = 1.5;
    float angle_threshold = pi;

    std::array<std::array<float, 6>, 4> *mat_a_device;
    std::array<float, 4> *vec_b_device;
    HANDLE_ERROR(cudaMalloc(&mat_a_device, sizeof(std::array<std::array<float, 6>, 4>)));
    HANDLE_ERROR(cudaMalloc(&vec_b_device, sizeof(std::array<float, 4>)));

    std::array<std::array<float, 6>, 4> true_mat_a = { { {-1.0,  1.0,  0.0, 0.0, 0.0, -1.0 },
                                                         { 0.0,  0.0,  0.0, 0.0, 0.0,  0.0 },
                                                         { 0.0,  0.0,  0.0, 0.0, 0.0,  0.0 },
                                                         { 0.0,  0.0,  0.0, 0.0, 0.0,  0.0 } } };
    std::array<float, 4> true_vec_b = { 1.0, 0.0, 0.0, 0.0 };

    std::cout.rdbuf(std::cerr.rdbuf());

    kernel::constructIcpResiduals(vertex_map, target_vertex_map, target_normal_map, no_rotation, no_translation,
        no_rotation, no_translation, intrinsics, distance_threshold, angle_threshold, (float *)mat_a_device, 
        (float*)vec_b_device);

    HANDLE_ERROR(cudaDeviceSynchronize());

    std::array<std::array<float, 6>, 4> mat_a;
    std::array<float, 4> vec_b;
    HANDLE_ERROR(cudaMemcpy(&mat_a, mat_a_device, sizeof(std::array<std::array<float, 6>, 4>), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(&vec_b, vec_b_device, sizeof(std::array<float, 4>), cudaMemcpyDeviceToHost));
    
    for (int i = 0; i < 4; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            ASSERT_FLOAT_EQ(true_mat_a[i][j], mat_a[i][j]);
        }
        ASSERT_FLOAT_EQ(true_vec_b[i], vec_b[i]);
    }
}

TEST_F(IcpTests, TestComputePose)
{
    // ####### Preparation ########
    const int width = 640;
    const int height = 480;

    cudaChannelFormatDesc raw_depth_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

    RgbdDataset rgbd_dataset;
    rgbd_dataset.load("../../rgbd_dataset_freiburg1_xyz");

    // ######### Frame 1 ###########
    DepthMap raw_depth_map_1(width, height, raw_depth_desc);
    auto next = rgbd_dataset.nextDepthAndPose();
    raw_depth_map_1.update(next.first);
    Sensor depth_sensor_1(525.0f);
    depth_sensor_1.setPose(next.second);
    CudaGridMap raw_depth_map_meters_1(width, height, depth_desc);
    kernel::convertToDepthMeters(raw_depth_map_1, raw_depth_map_meters_1, 1.0f / 5000.0f);
    
    GridMapPyramid<CudaGridMap> depth_map_pyramid_1(width, height, iters_per_layer.size(), depth_desc);
    kernel::applyBilateralFilter(raw_depth_map_meters_1, depth_map_pyramid_1[0]);
    kernel::downSample(depth_map_pyramid_1[0], depth_map_pyramid_1[1]);
    kernel::downSample(depth_map_pyramid_1[1], depth_map_pyramid_1[2]);

    GridMapPyramid<CudaGridMap> vertex_map_pyramid_1(width, height, iters_per_layer.size(), format_description);
    kernel::createVertexMap(depth_map_pyramid_1[0], vertex_map_pyramid_1[0], depth_sensor_1.getInverseIntr(0));
    kernel::createVertexMap(depth_map_pyramid_1[1], vertex_map_pyramid_1[1], depth_sensor_1.getInverseIntr(1));
    kernel::createVertexMap(depth_map_pyramid_1[2], vertex_map_pyramid_1[2], depth_sensor_1.getInverseIntr(2));

    GridMapPyramid<CudaGridMap> normal_map_pyramid_1(width, height, iters_per_layer.size(), format_description);
    kernel::createNormalMap(vertex_map_pyramid_1[0], normal_map_pyramid_1[0]);
    kernel::createNormalMap(vertex_map_pyramid_1[1], normal_map_pyramid_1[1]);
    kernel::createNormalMap(vertex_map_pyramid_1[2], normal_map_pyramid_1[2]);
    
    // ######### Save pose #########
    RigidTransform3D previous_pose;
    previous_pose.rot_mat = glm::mat3(1.0);
    previous_pose.transl_vec = glm::vec3(0.0);
    previous_pose.transl_vec = glm::vec3(0.0);
    
    //// ######### Frame 2 ###########
    DepthMap raw_depth_map_2(width, height, raw_depth_desc);
    next = rgbd_dataset.nextDepthAndPose();
    raw_depth_map_2.update(next.first);
    Sensor depth_sensor_2(525.0f);
    depth_sensor_2.setPose(next.second);
    
    CudaGridMap raw_depth_map_meters_2(width, height, depth_desc);
    kernel::convertToDepthMeters(raw_depth_map_2, raw_depth_map_meters_2, 1.0f / 5000.0f);
    
    GridMapPyramid<CudaGridMap> depth_map_pyramid_2(width, height, iters_per_layer.size(), depth_desc);
    kernel::applyBilateralFilter(raw_depth_map_meters_2, depth_map_pyramid_2[0]);
    kernel::downSample(depth_map_pyramid_2[0], depth_map_pyramid_2[1]);
    kernel::downSample(depth_map_pyramid_2[1], depth_map_pyramid_2[2]);
    
    GridMapPyramid<CudaGridMap> vertex_map_pyramid_2(width, height, iters_per_layer.size(), format_description);
    kernel::createVertexMap(depth_map_pyramid_2[0], vertex_map_pyramid_2[0], depth_sensor_2.getInverseIntr(0));
    kernel::createVertexMap(depth_map_pyramid_2[1], vertex_map_pyramid_2[1], depth_sensor_2.getInverseIntr(1));
    kernel::createVertexMap(depth_map_pyramid_2[2], vertex_map_pyramid_2[2], depth_sensor_2.getInverseIntr(2));
    
    // ######### Save pose #########
    RigidTransform3D true_pose;
    glm::mat4x4 homo = depth_sensor_1.getInversePose() * depth_sensor_2.getPose();
    true_pose.rot_mat = glm::mat3(homo);
    true_pose.transl_vec = homo[3];

    HANDLE_ERROR(cudaDeviceSynchronize());

    ICP icp(iters_per_layer, width, height, 0.1, pi / 3.0);
    RigidTransform3D pose_estimate = icp.computePose(vertex_map_pyramid_2, vertex_map_pyramid_1, normal_map_pyramid_1,
        previous_pose, depth_sensor_2);

    glm::vec3 v = glm::normalize(glm::vec3(1.0, 1.0, 1.0));
    glm::vec3 true_rotated_v = true_pose.rot_mat * v;
    glm::vec3 rotated_v = pose_estimate.rot_mat * v;
    float angle = glm::acos(glm::dot(true_rotated_v, rotated_v));

    ASSERT_LE(angle, pi / 90.0f);

    for (int i = 0; i < 3; i++)
    {
        ASSERT_NEAR(true_pose.transl_vec[i], pose_estimate.transl_vec[i], 0.01f);
    }
}