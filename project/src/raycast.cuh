#pragma once

#include "glm_macro.h"
#include "cuda_grid_map.h"
#include "voxel_grid.h"
#include "sensor.h"

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float raycast(const VoxelGridStruct& voxel_grid, const Sensor& sensor, const CudaGridMap& output_vertex, const CudaGridMap& output_normal, int level);
}