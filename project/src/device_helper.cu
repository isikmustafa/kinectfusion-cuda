#include "device_helper.cuh"

constexpr float VALID = -1.0f;
constexpr float INVALID = -2.0f;

namespace device_helper
{
	__device__ bool isDepthValid(float depth)
	{
		return depth > 0.0f && isfinite(depth);
	}
	__device__ void validate(cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		surf2Dwrite(VALID, grid_map, idx + 12, j);
	}

    __device__ void invalidate(cudaSurfaceObject_t grid_map, int i, int j)
    {
		int idx = i * 16;
        surf2Dwrite(INVALID, grid_map, idx + 12, j);
    }
    
    __device__ bool isValid(cudaSurfaceObject_t grid_map, int i, int j)
    {
		int idx = i * 16;
		float validity_indicator;
        surf2Dread(&validity_indicator, grid_map, idx + 12, j);
        
        return validity_indicator == VALID;
    }

	__device__ void writeVec3(const glm::vec3& vec3, cudaSurfaceObject_t grid_map, int i, int j)
	{
		int idx = i * 16;
		surf2Dwrite(vec3.x, grid_map, idx, j);
		surf2Dwrite(vec3.y, grid_map, idx + 4, j);
		surf2Dwrite(vec3.z, grid_map, idx + 8, j);
	}

	__device__ glm::vec3 computeNormal(cudaSurfaceObject_t grid_map, int i, int j)
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

		return glm::normalize(glm::cross(next_in_row - central_vertex, next_in_column - central_vertex));
	}
}