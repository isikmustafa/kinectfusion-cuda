#pragma once

#include "glm_macro.h"
#include <cuda_runtime.h>
#include <glm/mat3x3.hpp>

/*
    Helper struct to bundle a cuda surface object and a cuda array pointer that belong to each other
*/
struct CudaSurface
{
    cudaSurfaceObject_t surface_object{ 0 };
    cudaArray* cuda_array{ nullptr };
};