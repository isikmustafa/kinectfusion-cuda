#pragma once

#include "icp.h"
#include "cuda_utils.h"
#include "general_helper.h"
#include "icp.cuh"
#include "timer.h"

#include <glm/gtx/transform.hpp>
#include <glm/gtx/euler_angles.hpp>

ICP::ICP(const IcpConfig& config)
	: m_icp_config(config)
{
    HANDLE_ERROR(cudaMalloc(&m_mat_a, config.height * config.width * 6 * sizeof(float)));
    HANDLE_ERROR(cudaMalloc(&m_vec_b, config.height * config.width * sizeof(float)));
}

ICP::~ICP()
{
    cudaFree(m_mat_a);
    cudaFree(m_vec_b);
}

RigidTransform3D ICP::computePose(GridMapPyramid<CudaGridMap>& vertex_pyramid,
    GridMapPyramid<CudaGridMap>& target_vertex_pyramid, GridMapPyramid<CudaGridMap>& target_normal_pyramid, const Sensor& sensor)
{
	RigidTransform3D previous_pose(sensor.getPose());
	RigidTransform3D pose_estimate = previous_pose;
    m_execution_times = { 0.0f, 0.0f };

    for (int layer = m_icp_config.iters_per_layer.size() - 1; layer >= 0; layer--)
    {
        for (int i = 0; i < m_icp_config.iters_per_layer[layer]; i++)
        {
            m_execution_times[0] += kernel::constructIcpResiduals(vertex_pyramid[layer], target_vertex_pyramid[layer], 
                target_normal_pyramid[layer], previous_pose.rot_mat, previous_pose.transl_vec, pose_estimate.rot_mat, 
                pose_estimate.transl_vec, sensor.getIntr(layer), m_icp_config.distance_thresh, m_icp_config.angle_thresh, m_mat_a, m_vec_b);

			Timer timer;
            auto grid_dims = vertex_pyramid[layer].getGridDims();
			auto result = m_solver.solve(m_mat_a, m_vec_b, grid_dims[0] * grid_dims[1]);
			m_execution_times[1] += timer.getTime() * 1000.0;
    
			auto previous_estimate = pose_estimate;
            updatePose(pose_estimate, result);

            // If ICP does not make much difference, move directly to the next pyramid layer
			auto pose_error = poseError(previous_estimate.getTransformation(), pose_estimate.getTransformation());
            if (pose_error.first < m_icp_config.iteration_stop_thresh_angle && pose_error.second < m_icp_config.iteration_stop_thresh_distance)
            {
                break;
            }
        }
    }

	auto det = glm::determinant(pose_estimate.rot_mat);
	if (det< 0.98f || det > 1.001f)
	{
		std::cout << det << std::endl;
	}

    return pose_estimate;
}

std::array<float, 2> ICP::getExecutionTimes()
{
    return m_execution_times;
}

void ICP::updatePose(RigidTransform3D& pose, const std::array<float, 6>& result)
{
    for (int i = 0; i < 6; i++)
    {
        if (!std::isfinite(result[i]))
        {
            throw std::runtime_error("Error: an element in the result vector is not finite or NAN.");
        }
    }

    float beta = result[0];
    float gamma = result[1];
    float alpha = result[2];
    float t_x = result[3];
    float t_y = result[4];
    float t_z = result[5];

	glm::mat3 incremental_rotation = glm::orientate3(glm::vec3(beta, alpha, gamma));
    glm::vec3 incremental_translation(t_x, t_y, t_z);

    pose.rot_mat = glm::transpose(incremental_rotation) * pose.rot_mat;
    pose.transl_vec = glm::transpose(incremental_rotation) * pose.transl_vec + incremental_translation;
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