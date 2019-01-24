#pragma once
#include "icp.h"
#include <cusolverDn.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

#include "cuda_utils.h"
#include "measurement.cuh"
#include "icp.cuh"

ICP::ICP(std::vector<unsigned int> iters_per_layer, unsigned int width, unsigned int height, float distance_thresh, 
    float angle_thresh)
    : m_distance_thresh(distance_thresh)
    , m_iters_per_layer(iters_per_layer)
    , m_angle_thresh(angle_thresh)
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
    GridMapPyramid<CudaGridMap> &target_vertex_pyramid, GridMapPyramid<CudaGridMap> &target_normal_pyramid,
    RigidTransform3D &previous_pose, Sensor sensor)
{
    // Initialize pose estimate to current one
    RigidTransform3D pose_estimate = previous_pose;
	m_execution_times = { 0.0f, 0.0f };

    for (int layer = m_iters_per_layer.size() - 1; layer >= 0; layer--)
    {
        for (int i = 0; i < m_iters_per_layer[layer]; i++)
        {
            m_execution_times[0] += kernel::constructIcpResiduals(vertex_pyramid[layer], target_vertex_pyramid[layer], 
                target_normal_pyramid[layer], previous_pose.rot_mat, previous_pose.transl_vec, pose_estimate.rot_mat, 
                pose_estimate.transl_vec, sensor.getIntr(layer), m_distance_thresh, m_angle_thresh, m_mat_a, m_vec_b);

            auto grid_dims = vertex_pyramid[layer].getGridDims();
            m_execution_times[1] += solver.solve(m_mat_a, m_vec_b, grid_dims[0] * grid_dims[1], m_vec_x);
    
            updatePose(pose_estimate);
        }
    }

    return pose_estimate;
}

std::array<float, 2> ICP::getExecutionTimes()
{
    return m_execution_times;
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

    if (isnan(alpha) || isnan(beta) || isnan(gamma) || isnan(t_x), isnan(t_y), isnan(t_z))
    {
        throw std::runtime_error("Error: NANs in result vector x.");
    }

    glm::mat3x3 incremental_rotation(
          glm::rotate(alpha, glm::vec3(0.0f, 0.0f, 1.0f))
        * glm::rotate(gamma, glm::vec3(0.0f, 1.0f, 0.0f))
        * glm::rotate(beta, glm::vec3(1.0f, 0.0f, 0.0f)));
    glm::vec3 incremental_translation(t_x, t_y, t_z);

    pose.rot_mat = glm::transpose(incremental_rotation) * pose.rot_mat;
    pose.transl_vec = glm::transpose(incremental_rotation) * pose.transl_vec + incremental_translation;
}
