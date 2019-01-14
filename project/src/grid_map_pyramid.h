#pragma once

#include <vector>
#include <memory>

/*
    Container class for grid map pyramids. 
    Main purpose is the convenient creation of such pyramids, common operatinos on pyramids and capsuled memory management.
*/
template<class T>
class GridMapPyramid
{
public:
    GridMapPyramid(unsigned int width, unsigned int height, unsigned int n_layers, 
        cudaChannelFormatDesc channel_description)
		: m_base_width(width)
		, m_base_height(height)
		, m_n_layers(n_layers)
        , m_channel_description(channel_description)
        , m_pyramid(n_layers)
    {
		checkGridDimensions();

        for (int i = 0; i < m_n_layers; i++)
        {
            m_pyramid[i] = std::make_unique<T>(m_base_width >> i, m_base_height >> i, m_channel_description);
        }
    }

	T& operator[](int i)
    {
        return *(m_pyramid[i]);
    }

	const T& operator[](int i) const
	{
		return *(m_pyramid[i]);
	}
    
    cudaChannelFormatDesc getChannelDescription() const
    {
        return m_channel_description;
    }

    unsigned int getBaseWidth() const
    {
        return m_base_width;
    }

    unsigned int getBaseHeight() const
    {
        return m_base_height;
    }

private:
	std::vector<std::unique_ptr<T>> m_pyramid;
	cudaChannelFormatDesc m_channel_description;
    unsigned int m_base_width;
    unsigned int m_base_height;
    unsigned int m_n_layers;

private:
    void checkGridDimensions()
    {
		auto layer_exp = 1 << (m_n_layers - 1);

        bool cannot_construct_pyramid = ((m_base_width % layer_exp) != 0) || ((m_base_height % layer_exp) != 0);
        if (cannot_construct_pyramid)
        {
            throw std::runtime_error{ "Can't construct pyramid. Dimensions must be divisible by 2^(m_n_layers - 1)." };
        }
    }
};

