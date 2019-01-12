#pragma once

#include <array>
#include <memory>

/*
    Container class for grid map pyramids. 
    Main purpose is the convenient creation of such pyramids, common operatinos on pyramids and capsuled memory management.
*/

template<class T, unsigned int tNumLayers = 3>
class GridMapPyramid
{
public:
    GridMapPyramid(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
		: m_channel_description(channel_description)
		, m_base_width(width)
		, m_base_height(height)
    {
		checkGridDimensions(width, height);

        for (int i = 0; i < tNumLayers; i++)
        {
            m_pyramid[i] = std::make_unique<T>(m_base_width >> i, m_base_height >> i, m_channel_description);
        }
    }

    virtual T& GridMapPyramid::operator[](int i)
    {
        return *(m_pyramid[i]);
    }
    
    cudaChannelFormatDesc getChannelDescription()
    {
        return m_channel_description;
    }

    unsigned int getBaseWidth()
    {
        return m_base_width;
    }

    unsigned int getBaseHeight()
    {
        return m_base_height;
    }

private:
	std::array<std::unique_ptr<T>, tNumLayers> m_pyramid;
	cudaChannelFormatDesc m_channel_description;
    unsigned int m_base_width;
    unsigned int m_base_height;

private:
    void checkGridDimensions(unsigned int width, unsigned int height)
    {
		constexpr auto layer_exp = 1 << (tNumLayers - 1);

        bool cannot_construct_pyramid = ((width % layer_exp) != 0) || ((height % layer_exp) != 0);
        if (cannot_construct_pyramid)
        {
            throw std::runtime_error{ "Can't construct pyramid. Dimensions must be divisible by 2^(tNumLayers - 1)." };
        }
    }
};

