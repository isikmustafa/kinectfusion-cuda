#pragma once

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "depth_map.h"
#include "cuda_utils.h"

DepthMap::DepthMap(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
	: CudaGridMap(width, height, channel_description)
{};

void DepthMap::update(const std::string& path)
{
	int channel;
	int width;
	int height;
	auto data = stbi_load_16(path.c_str(), &width, &height, &channel, 0);
	if (!data)
	{
		throw std::runtime_error("Error: Image cannot be loaded");
	}

    update(data);
	stbi_image_free(data);
}

void DepthMap::update(void* data_ptr)
{
    int n_bytes = m_channel_description.x / 8;
	HANDLE_ERROR(cudaMemcpyToArray(m_grid_elems.cuda_array, 0, 0, 
        data_ptr, m_width * m_height * n_bytes, cudaMemcpyHostToDevice));
}