#pragma once
#include "cuda_grid_map.h"

#include <string>
#include <iostream>

#include <stb_image_write.h>
#include "cuda_utils.h"

CudaGridMap::CudaGridMap(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
{
    m_width = width;
    m_height = height;
    m_channel_description = channel_description;

	//Allocate arrays.
	HANDLE_ERROR(cudaMallocArray(&m_grid_elems.cuda_array, &channel_description, width, height, cudaArraySurfaceLoadStore));

	//Create resource descriptions.
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	//Create CUDA Surface objects
	res_desc.res.array.array = m_grid_elems.cuda_array;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_grid_elems.surface_object, &res_desc));
}

CudaGridMap::~CudaGridMap()
{
	HANDLE_ERROR(cudaDestroySurfaceObject(m_grid_elems.surface_object));
	HANDLE_ERROR(cudaFreeArray(m_grid_elems.cuda_array));
}

std::array<unsigned int, 2> CudaGridMap::getGridDims() const
{
    return { m_width, m_height };
}

cudaSurfaceObject_t CudaGridMap::getCudaSurfaceObject() const
{
    return m_grid_elems.surface_object;
}

cudaArray* CudaGridMap::getCudaArray() const
{
    return m_grid_elems.cuda_array;
}

cudaChannelFormatDesc CudaGridMap::getChannelDescription() const
{
    return m_channel_description;
}