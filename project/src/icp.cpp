#pragma once
#include "icp.h"
#include "data_helper.h"
#include "measurement.cuh"

ICP::ICP(unsigned int grid_width, unsigned int grid_height, cudaChannelFormatDesc format_description, 
    std::vector<unsigned int> iters_per_layer, float distance_thresh, float angle_thresh)
    : m_distance_thresh(distance_thresh)
    , m_angle_thresh(angle_thresh)
    , m_transformed_vertex_pyramid(grid_width, grid_height, format_description)
    , m_normal_pyramid(grid_width, grid_height, format_description)
    , m_target_normal_pyramid(grid_width, grid_height, format_description)
{
}

ICP::~ICP()
{
}
