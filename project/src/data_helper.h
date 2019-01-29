#pragma once

#include "glm_macro.h"
#include <cuda_runtime.h>
#include <glm/glm.hpp>

/*
    Helper struct to bundle a cuda surface object and a cuda array pointer that belong to each other
*/
struct CudaSurface
{
    cudaSurfaceObject_t surface_object{ 0 };
    cudaArray* cuda_array{ nullptr };
};

typedef glm::ivec2 Coords2D;
typedef glm::ivec3 Coords3D;