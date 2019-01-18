#include "device_helper.cuh"

namespace device_helper
{
    constexpr float cValid = -1.0f;
    constexpr float cInvalid = -2.0f;

    // Checks if given depth value with float precision is considered as valid or not.
    __device__ bool isDepthValid(float depth)
    {
        return depth > 0.0f && isfinite(depth);
    }

    // Validates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
    // where the 4th element has to be unused.
    __device__ void validate(cudaSurfaceObject_t grid_map, int i, int j)
    {
        int idx = i * 16;
        surf2Dwrite(cValid, grid_map, idx + 12, j);
    }

    // Invalidates the element (i, j) of the grid map. The function expects a gridmap with 4D vectors as elements, 
    // where the 4th element has to be unused.
    __device__ void invalidate(cudaSurfaceObject_t grid_map, int i, int j)
    {
        int idx = i * 16;
        surf2Dwrite(cInvalid, grid_map, idx + 12, j);
    }

    // Check whether the element (i, j) of a grid map is valid. The function expects a gridmap with 4D vectors as 
    // elements, where the 4th element has to be unused.
    __device__ bool isValid(cudaSurfaceObject_t grid_map, int i, int j)
    {
        int idx = i * 16;
        float validity_indicator;
        surf2Dread(&validity_indicator, grid_map, idx + 12, j);

        return validity_indicator == cValid;
    }

    // Writes vec3 with float members to grid_map. The function expects a gridmap with 4D vectors as 
    // elements, where the 4th element has to be unused.
    __device__ void writeVec3(const glm::vec3& vec3, cudaSurfaceObject_t grid_map, int i, int j)
    {
        int idx = i * 16;
        surf2Dwrite(vec3.x, grid_map, idx, j);
        surf2Dwrite(vec3.y, grid_map, idx + 4, j);
        surf2Dwrite(vec3.z, grid_map, idx + 8, j);
    }

    // Computes the normal of surface corresponding to (i, j) with finite difference method. The function
    // expects a gridmap with 4D vectors as elements, where the 4th element has to be unused.
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