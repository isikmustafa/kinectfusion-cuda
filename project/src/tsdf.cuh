#pragma once

#include "glm_macro.h"

#include "voxel_grid.h"
#include "sensor.h"

namespace kernel
{
	//As paper suggests, these are approximately the conservative minimum and conservative maximum sensor ranges.
	constexpr float cMinDistance = 0.4f;
	constexpr float cMaxDistance = 8.0f;

	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float fuse(cudaSurfaceObject_t raw_depth_map_meters, const VoxelGridStruct& voxel_grid, const Sensor& sensor);
}