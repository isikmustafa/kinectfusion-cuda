#pragma once

#include "cuda_grid_map.h"

#include <string>
#include <cuda_runtime.h>

class DepthMap : public CudaGridMap
{
public:
	DepthMap(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description);

	void update(const std::string& path);
	void update(void* data_ptr);
};