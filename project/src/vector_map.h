#pragma once

#include <array>

#include <cuda_runtime.h>
#include "data_helper.h"

/*
    Base class containing common functionality for any vector maps, e.g. vertex maps or normal maps.
    An object contains a set of vectors, each one mapped to a different cell on a 2D grid.
    Vectors are stored as CUDA surface objects.
*/
class VectorMap
{
public:
	VectorMap();
	~VectorMap();

	std::array<cudaSurfaceObject_t, 3> getPyramid() const;
    
    // Just for debugging.
	void writePyramidToFile(std::string file_name) const;

private:

    cuda_surface m_vectors_640x480;
    cuda_surface m_vectors_320x240;
    cuda_surface m_vectors_160x120;

private:
    // Just for debugging.
	void writeSurface(std::string file_name, cudaArray* gpu_source, int width, int height) const;
};