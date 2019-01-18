#pragma once
#include "icp.h"
#include <cusolverDn.h>

ICP::ICP(RigidTransform3D previous_transform, std::vector<unsigned int> iters_per_layer, unsigned int width, 
    unsigned int height, float distance_thresh, float angle_thresh)
    : m_previous_transform(previous_transform)
    , m_distance_thresh(distance_thresh)
    , m_angle_thresh(angle_thresh)
    , m_normal_format_description{ cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat) }
    , m_target_normal_pyramid(width, height, iters_per_layer.size(), m_normal_format_description)
{
}

ICP::~ICP()
{
    /*
        TODO:
        1. Compute target normal pyramid
        2. Initialize transformation
        3. 
    */
}

RigidTransform3D ICP::computePose(GridMapPyramid<CudaGridMap>& vertex_pyramid, GridMapPyramid<CudaGridMap>& target_vertex_pyramid, RigidTransform3D & previous_pose)
{
    return RigidTransform3D();
}
