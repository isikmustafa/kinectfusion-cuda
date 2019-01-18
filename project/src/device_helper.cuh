#pragma once

#include <cuda_runtime.h>
#include "glm_macro.h"
#include <glm/glm.hpp>

namespace device_helper
{
	// Checks if given depth value with float precision is considered as valid or not.
    __device__ bool isDepthValid(float depth);

	// Validates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
	// where the 4th element has to be unused.
    __device__ void validate(cudaSurfaceObject_t grid_map, int i, int j);

	// Invalidates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
	// where the 4th element has to be unused.
    __device__ void invalidate(cudaSurfaceObject_t grid_map, int i, int j);

	// Check whether the element (i, j) of a grid map is valid. The function expects a gridmap with 4D vectors as 
	// elements, where the 4th element has to be unused.
    __device__ bool isValid(cudaSurfaceObject_t grid_map, int i, int j);

	// Writes vec3 with float members to grid_map. The function expects a gridmap with 4D vectors as 
	// elements, where the 4th element has to be unused.
    __device__ void writeVec3(const glm::vec3& vec3, cudaSurfaceObject_t grid_map, int i, int j);

	// Computes the normal of surface corresponding to (i, j) with finite difference method. The function
	// expects a gridmap with 4D vectors as elements, where the 4th element has to be unused.
    __device__ glm::vec3 computeNormal(cudaSurfaceObject_t grid_map, int i, int j);
}