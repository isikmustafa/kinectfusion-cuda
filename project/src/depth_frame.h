#pragma once

#include <string>
#include <array>

#include <cuda_runtime.h>
#include "data_helper.h"

class DepthFrame
{
public:
	DepthFrame();
	~DepthFrame();

	void update(const std::string& path);
	void update(void* data_ptr);
	cudaSurfaceObject_t getRaw() const;
	std::array<cudaSurfaceObject_t, 3> getPyramid() const;
	void writePyramid() const;

private:
    cuda_surface m_depth_raw;
    cuda_surface m_depth_640x480;
    cuda_surface m_depth_320x240;
    cuda_surface m_depth_160x120;

private:
	void writeSurface(cudaArray* gpu_source, int width, int height) const;
};