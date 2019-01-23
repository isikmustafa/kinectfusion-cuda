#pragma once
#include "icp.h"
#include <cusolverDn.h>

#include "cuda_utils.h"
#include "measurement.cuh"
#include "icp.cuh"

ICP::ICP(std::vector<unsigned int> iters_per_layer, unsigned int width, unsigned int height, float distance_thresh,
    float angle_thresh, glm::mat3x3 sensor_intrinsics)
    : m_distance_thresh(distance_thresh)
    , m_iters_per_layer(iters_per_layer)
    , m_angle_thresh(angle_thresh)
    , m_normal_format_description{ cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat) }
    , m_target_normal_pyramid(width, height, iters_per_layer.size(), m_normal_format_description)
    , m_sensor_intrinsics(sensor_intrinsics)
{
    HANDLE_ERROR(cudaMalloc(&m_mat_a, height * width * 6 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&m_vec_b, height * width * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&m_vec_x, 6 * sizeof(float)));
}

ICP::~ICP()
{
    cudaFree(m_mat_a);
    cudaFree(m_vec_b);
    cudaFree(m_vec_x);
}

RigidTransform3D ICP::computePose(GridMapPyramid<CudaGridMap> &vertex_pyramid, 
    GridMapPyramid<CudaGridMap> &target_vertex_pyramid, RigidTransform3D &previous_pose)
{
    // Create target normal pyramid
    kernel::computeNormalMap(vertex_pyramid[0], m_target_normal_pyramid[0]);
    kernel::computeNormalMap(vertex_pyramid[1], m_target_normal_pyramid[1]);
    kernel::computeNormalMap(vertex_pyramid[2], m_target_normal_pyramid[2]);

    // Initialize pose estimate to current one
    RigidTransform3D pose_estimate = previous_pose;

    //for (int layer = m_iters_per_layer.size() - 1; layer > 0; layer--)
    //{
    //    for (int i = 0; i < m_iters_per_layer[layer]; i++)
    //    {
    //        kernel::constructIcpResiduals(vertex_pyramid[layer], target_vertex_pyramid[layer], 
    //            m_target_normal_pyramid[layer], previous_pose.rot_mat, previous_pose.transl_vec, pose_estimate.rot_mat, 
    //            pose_estimate.transl_vec, m_sensor_intrinsics, m_distance_thresh, m_angle_thresh, m_mat_a, m_vec_b);
    //
    //        auto grid_dims = vertex_pyramid[layer].getGridDims();
    //        solver.solve(m_mat_a, m_vec_b, grid_dims[0] * grid_dims[1], m_vec_x);
    //
    //        updatePose(pose_estimate);
    //    }
    //}

    for (int i = 0; i < m_iters_per_layer[0]; i++)
        {
            kernel::constructIcpResiduals(vertex_pyramid[0], target_vertex_pyramid[0], 
                m_target_normal_pyramid[0], previous_pose.rot_mat, previous_pose.transl_vec, pose_estimate.rot_mat, 
                pose_estimate.transl_vec, m_sensor_intrinsics, m_distance_thresh, m_angle_thresh, m_mat_a, m_vec_b);
        
            auto grid_dims = vertex_pyramid[0].getGridDims();
            solver.solve(m_mat_a, m_vec_b, grid_dims[0] * grid_dims[1], m_vec_x);
        
            updatePose(pose_estimate);
        }

    return pose_estimate;
}

void ICP::updatePose(RigidTransform3D &pose)
{
    std::array<float, 6> vec_x_host;
    HANDLE_ERROR(cudaMemcpy(&vec_x_host, m_vec_x, 6 * sizeof(float), cudaMemcpyDeviceToHost));

    float beta = vec_x_host[0];
    float gamma = vec_x_host[1];
    float alpha = vec_x_host[2];
    float t_x = vec_x_host[3];
    float t_y = vec_x_host[4];
    float t_z = vec_x_host[5];

    glm::mat3x3 incremental_rotation(
        glm::vec3(  1.0f, -alpha, gamma),
        glm::vec3( alpha,  1.0f,  -beta),
        glm::vec3(-gamma, -beta,  1.0f));
    glm::vec3 incremental_translation(t_x, t_y, t_z);

    pose.rot_mat = incremental_rotation * pose.rot_mat;
    pose.transl_vec = incremental_translation + pose.transl_vec;
}
