#pragma once

#include <array>

#include <cuda_runtime.h>

class VertexMap
{
public:
	VertexMap();
	~VertexMap();

	std::array<cudaSurfaceObject_t, 3> getPyramid() const;
	void writePyramid() const;

private:
	cudaSurfaceObject_t m_vertex_640x480{ 0 };
	cudaSurfaceObject_t m_vertex_320x240{ 0 };
	cudaSurfaceObject_t m_vertex_160x120{ 0 };
	cudaArray* m_cuda_array_640x480{ nullptr };
	cudaArray* m_cuda_array_320x240{ nullptr };
	cudaArray* m_cuda_array_160x120{ nullptr };

private:
	void writeSurface(cudaArray* gpu_source, int width, int height) const;
};