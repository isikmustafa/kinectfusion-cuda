#pragma once

#include "glm_macro.h"
#include "cuda_grid_map.h"
#include "voxel_grid.h"
#include "sensor.h"

namespace kernel
{
	//As paper suggests, these are approximately the conservative minimum and conservative maximum sensor ranges.
	constexpr float cMinDistance = 0.4f;
	constexpr float cMaxDistance = 6.0f;

	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float initializeGrid(const VoxelGridStruct& voxel_grid, const Voxel& value);
	float fuse(const CudaGridMap& raw_depth_map_meters, const VoxelGridStruct& voxel_grid, const Sensor& sensor);
}