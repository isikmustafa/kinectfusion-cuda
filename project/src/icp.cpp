#pragma once
#include "icp.h"
#include "data_helper.h"
#include "measurement.cuh"

ICP::ICP(GridMapPyramid<CudaGridMap> &vertex_pyramid, GridMapPyramid<DepthMap> &depth_predicton_pyramid, RigidTransform3D &previous_pose,
    int n_iterations, float distance_thresh, float angle_thresh)
{
    m_n_iterations = n_iterations;
    m_distance_thresh = distance_thresh;
    m_angle_thresh = angle_thresh;
    m_previous_pose = previous_pose;
    
    m_pose = RigidTransform3D();

    m_vertex_pyramid = &vertex_pyramid;
    m_depth_prediction_pyramid = &depth_predicton_pyramid;
    m_normal_pyramid = new GridMapPyramid<CudaGridMap>(vertex_pyramid.getBaseWidth(), vertex_pyramid.getBaseHeight(), vertex_pyramid.getChannelDescription());

    for (int i = 0; i < 3; i++)
    {
        kernel::computeNormalMap((*m_vertex_pyramid)[i], (*m_normal_pyramid)[i]);
    }
}

ICP::~ICP() 
{
    delete m_normal_pyramid;
}

void ICP::initializePose(glm::mat4x4 initial_pose)
{
}
