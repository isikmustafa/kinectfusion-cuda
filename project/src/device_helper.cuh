#pragma once

#include "glm_macro.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace device_helper
{
	constexpr float cValid = -1.0f;
	constexpr float cInvalid = -2.0f;
    constexpr float cMinDepth = 0.1f;

	// Checks if given depth value with float precision is considered as valid or not.
	__device__ inline bool isDepthValid(float depth)
	{
		return depth > cMinDepth && isfinite(depth);
	}

	// Validates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
	// where the 4th element has to be unused.
	__device__ inline void validate(cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		surf2Dwrite(cValid, grid_map, idx + 12, j);
	}

	// Invalidates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
	// where the 4th element has to be unused.
	__device__ inline void invalidate(cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		surf2Dwrite(cInvalid, grid_map, idx + 12, j);
	}

	// Check whether the element (i, j) of a grid map is valid. The function expects a gridmap with 4D vectors as 
	// elements, where the 4th element has to be unused.
	__device__ inline bool isValid(cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		float validity_indicator;
		surf2Dread(&validity_indicator, grid_map, idx + 12, j, cudaBoundaryModeZero);

		return validity_indicator == cValid;
	}

	// Writes vec3 with float members to grid_map. The function expects a gridmap with 4D vectors as 
	// elements, where the 4th element has to be unused.
	__device__ inline void writeVec3(const glm::vec3& vec3, cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		surf2Dwrite(vec3.x, grid_map, idx, j);
		surf2Dwrite(vec3.y, grid_map, idx + 4, j);
		surf2Dwrite(vec3.z, grid_map, idx + 8, j);
	}

    // Writes vec4 with float members to grid_map. The function expects a gridmap with 4D vectors as 
    // elements, where the 4th element has to be unused.
    __device__ inline void writeVec4(const glm::vec4& vec4, cudaSurfaceObject_t grid_map, int i, int j)
    {
        int idx = i * 16;
        surf2Dwrite(vec4.x, grid_map, idx, j);
        surf2Dwrite(vec4.y, grid_map, idx + 4, j);
        surf2Dwrite(vec4.z, grid_map, idx + 8, j);
        surf2Dwrite(vec4.w, grid_map, idx + 12, j);
    }

	// Reads vec3 with float members from grid_map. The function expects a gridmap with 4D vectors as 
	// elements, where the 4th element has to be unused.
	__device__ inline glm::vec3 readVec3(cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		glm::vec3 vec3;
		surf2Dread(&vec3.x, grid_map, idx, j);
		surf2Dread(&vec3.y, grid_map, idx + 4, j);
		surf2Dread(&vec3.z, grid_map, idx + 8, j);
		return vec3;
	}

	// Computes the normal of surface corresponding to (i, j) with finite difference method. The function
	// expects a gridmap with 4D vectors as elements, where the 4th element has to be unused.
	__device__ inline glm::vec4 computeNormal(cudaSurfaceObject_t grid_map, int i, int j)
	{   
        glm::vec4 normal;
        if (isValid(grid_map, i, j) && isValid(grid_map, i + 1, j) && isValid(grid_map, i, j + 1))
        {
            int idx = i * 16;

            glm::vec3 central_vertex, next_in_row, next_in_column;
            surf2Dread(&central_vertex.x, grid_map, idx, j);
            surf2Dread(&central_vertex.y, grid_map, idx + 4, j);
            surf2Dread(&central_vertex.z, grid_map, idx + 8, j);

            surf2Dread(&next_in_row.x, grid_map, idx + 16, j, cudaBoundaryModeClamp);
            surf2Dread(&next_in_row.y, grid_map, idx + 20, j, cudaBoundaryModeClamp);
            surf2Dread(&next_in_row.z, grid_map, idx + 24, j, cudaBoundaryModeClamp);

            surf2Dread(&next_in_column.x, grid_map, idx, j + 1, cudaBoundaryModeClamp);
            surf2Dread(&next_in_column.y, grid_map, idx + 4, j + 1, cudaBoundaryModeClamp);
            surf2Dread(&next_in_column.z, grid_map, idx + 8, j + 1, cudaBoundaryModeClamp);

            normal = glm::vec4(glm::normalize(glm::cross(next_in_row - central_vertex, next_in_column - central_vertex)),
                cValid);
        }
        else
        {
            normal = glm::vec4(0.0f, 0.0f, 0.0f, cInvalid);
        }

        return normal;
	}
}