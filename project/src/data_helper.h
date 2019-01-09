#pragma once

#include <cuda_runtime.h>

/*
    Helper struct to bundle a cuda surface object and a cuda array pointer that belong to each other
*/
struct cuda_surface
{
    cudaSurfaceObject_t surface_object{ 0 };
    cudaArray* cuda_array{ nullptr };
};