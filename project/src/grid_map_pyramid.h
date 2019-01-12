#pragma once
#include <array>

#include "cuda_grid_map.h"

/*
    Container class for grid map pyramids. 
    Main purpose is the convenient creation of such pyramids, common operatinos on pyramids and capsuled memory management.
*/
template<class T>
class GridMapPyramid
{
public:
    GridMapPyramid(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
    {
        m_base_width = width;
        m_base_height = height;
        m_channel_description = channel_description;

        checkGridDimenstions(width, height);

        int factor = 1;
        for (int i = 0; i < m_n_layers; i++)
        {
            m_pyramid[i] = new T(m_base_width / factor, m_base_height / factor, m_channel_description);
            factor *= 2;
        }
    }

    ~GridMapPyramid() 
    {
        for (const auto map : m_pyramid)
        {
            delete map;
        }
    }

    virtual T& GridMapPyramid::operator[](int i)
    {
        return *(m_pyramid[i]);
    }
    
    cudaChannelFormatDesc get_channel_description()
    {
        return m_channel_description;
    }

    void foo();

private:
    const unsigned int m_n_layers = 3;
    unsigned int m_base_width;
    unsigned int m_base_height;
    cudaChannelFormatDesc m_channel_description;
    std::array<T*, 3> m_pyramid;

    void checkGridDimenstions(unsigned int width, unsigned int height)
    {
        bool cannot_construct_pyramid = ((width % 4) != 0) || ((height % 4) != 0);
        if (cannot_construct_pyramid)
        {
            throw std::runtime_error{ "Can't construct pyramid. Dimensions must be divisible by 4." };
        }
    }
};

