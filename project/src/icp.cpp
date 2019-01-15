#pragma once
#include "icp.h"

ICP::ICP(std::vector<unsigned int> iters_per_layer, float distance_thresh, float angle_thresh)
    : m_distance_thresh(distance_thresh)
    , m_angle_thresh(angle_thresh)
{
}

ICP::~ICP()
{
}
