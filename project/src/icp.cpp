#pragma once
#include "icp.h"
#include <glm/gtx/transform.hpp>

#include "cuda_utils.h"
#include "icp.cuh"

ICP::ICP(IcpConfig &config)
    : m_distance_thresh(config.distance_thresh)
    , m_iters_per_layer(config.iters_per_layer)
    , m_angle_thresh(config.angle_thresh)
{
    HANDLE_ERROR(cudaMalloc(&m_mat_a, config.height * config.width * m_n_variables * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&m_vec_b, config.height * config.width * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&m_vec_x, m_n_variables * sizeof(float)));
}

ICP::~ICP()
{
    cudaFree(m_mat_a);
    cudaFree(m_vec_b);
    cudaFree(m_vec_x);
}

RigidTransform3D ICP::computePose(GridMapPyramid<CudaGridMap> &vertex_pyramid,
    GridMapPyramid<CudaGridMap> &target_vertex_pyramid, GridMapPyramid<CudaGridMap> &target_normal_pyramid,
    Sensor sensor)
{
    RigidTransform3D previous_pose, pose_estimate;
    previous_pose.rot_mat = glm::mat3x3(sensor.getPose());
    previous_pose.transl_vec = sensor.getPosition();
    pose_estimate = previous_pose;

    m_execution_times = { 0.0f, 0.0f };
	int counter = 0;//*
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

			counter++;// match it to frame number

            // If ICP converged, move directly to the next pyramid layer
            if (solver.getLastError() < m_stop_thresh)
            {
                break;
            }
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
    HANDLE_ERROR(cudaMemcpy(&vec_x_host, m_vec_x, m_n_variables * sizeof(float), cudaMemcpyDeviceToHost));

    for (int i = 0; i < m_n_variables; i++)
    {
        if (!std::isfinite(vec_x_host[i]))
        {
            throw std::runtime_error("Error: an element in the result vector is not finite or NAN.");
        }
    }

    float beta = vec_x_host[0];
    float gamma = vec_x_host[1];
    float alpha = vec_x_host[2];
    float t_x = vec_x_host[3];
    float t_y = vec_x_host[4];
    float t_z = vec_x_host[5];

    glm::mat3x3 incremental_rotation = buildRotationZYX(alpha, gamma, beta);
    glm::vec3 incremental_translation(t_x, t_y, t_z);

    pose.rot_mat = glm::transpose(incremental_rotation) * pose.rot_mat;
    pose.transl_vec = glm::transpose(incremental_rotation) * pose.transl_vec + incremental_translation;
}

glm::mat3x3 ICP::buildRotationZYX(float z_angle, float y_angle, float x_angle)
{
    return  glm::mat3x3(glm::rotate(z_angle, glm::vec3(0.0f, 0.0f, 1.0f))
                      * glm::rotate(y_angle, glm::vec3(0.0f, 1.0f, 0.0f))
                      * glm::rotate(x_angle, glm::vec3(1.0f, 0.0f, 0.0f)));
}

unsigned int ICP::countResiduals(unsigned int max_idx)
{
    std::vector<float> temp(max_idx);
    HANDLE_ERROR(cudaMemcpy(&(temp[0]), m_vec_b, sizeof(float) * max_idx, cudaMemcpyDeviceToHost));

    unsigned int counter = 0;
    for (int i = 0; i < max_idx; i++)
    {
        if (temp[i] != 0.0f)
        {
            counter++;
        }
    }
    return counter;
}
