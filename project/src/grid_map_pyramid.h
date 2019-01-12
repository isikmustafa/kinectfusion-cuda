#pragma once
#include <array>

#include "cuda_grid_map.h"
#include "depth_map.h"

/*
    Container class for grid map pyramids. 
    Main purpose is the convenient creation of such pyramids, common operatinos on pyramids and capsuled memory management.
*/
class GridMapPyramid
{
public:
    GridMapPyramid(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description);
    ~GridMapPyramid();

    CudaGridMap& GridMapPyramid::operator[](int i);
    cudaChannelFormatDesc get_channel_description();

protected:
    // possible extension: choose arbitrary layer count
    const unsigned int m_n_layers = 3;
    unsigned int m_base_width;
    unsigned int m_base_height;
    cudaChannelFormatDesc m_channel_description;
    std::array<CudaGridMap*, 3> m_pyramid;
};

