#pragma once
#include <cuda_runtime.h>
#include "glm_macro.h"
#include <glm/mat3x3.hpp>

#include "cuda_grid_map.h"
#include "rigid_transform_3D.h"

namespace device_helpers
{
    // Invalidates the element (u, v) of the grid map. The function expects a gridmap with 4D vectors as elements, 
    // where the 4th element has to be unused.
    __device__ void invalidate(cudaSurfaceObject_t grid_map, unsigned int u, unsigned int v);

    // Check whether the element (u, v) of a grid map is valid. The function expects a gridmap with 4D vectors as 
    //elements, where the 4th element has to be unused.
    __device__ bool is_invalid(cudaSurfaceObject_t grid_map, unsigned int u, unsigned int v);
}