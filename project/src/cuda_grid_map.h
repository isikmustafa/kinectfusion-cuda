#pragma once

#include <array>

#include <cuda_runtime.h>
#include "data_helper.h"

/*
    Base class containing common functionality for any vector maps, e.g. vertex maps or normal maps.
    An object contains a set of vectors, each one mapped to a different cell on a 2D grid.
    Vectors are stored as CUDA surface objects.
*/
class CudaGridMap
{
public:
	CudaGridMap(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description);
	~CudaGridMap();

    std::array<unsigned int, 2> getGridDims() const;
    cudaSurfaceObject_t getCudaSurfaceObject() const;
    cudaArray* getCudaArray() const;
    cudaChannelFormatDesc getChannelDescription() const;

protected:
    unsigned int m_width;
    unsigned int m_height;
    cudaChannelFormatDesc m_channel_description;
    CudaSurface m_grid_elems;
};