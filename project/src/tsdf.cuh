#include "glm_macro.h"

#include "voxel_grid.h"
#include "sensor.h"

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float fuse(cudaSurfaceObject_t raw_depth_map, const VoxelGrid& voxel_grid, const Sensor& sensor);
}