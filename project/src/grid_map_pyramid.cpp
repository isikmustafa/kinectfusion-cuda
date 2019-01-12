#include "grid_map_pyramid.h"


GridMapPyramid::GridMapPyramid(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
{
    m_base_width = width;
    m_base_height = height;
    m_channel_description = channel_description;

    bool cannot_construct_pyramid = ((width % 4) != 0) || ((height % 4) != 0);
    if (cannot_construct_pyramid)
    {
        throw std::runtime_error{ "Can't construct pyramid. Dimensions must be divisible by 4." };
    }

    int factor = 1;
    for (int i = 0; i < m_n_layers; i++)
    {
        m_pyramid[i] = new CudaGridMap(width / factor, height / factor, channel_description);
        factor *= 2;
    }
}

GridMapPyramid::~GridMapPyramid()
{
    for (const auto map : m_pyramid)
    {
        delete map;
    }
}

CudaGridMap& GridMapPyramid::operator[](int i) {
    return *(m_pyramid[i]);
}

cudaChannelFormatDesc GridMapPyramid::get_channel_description()
{
    return m_channel_description;
}
