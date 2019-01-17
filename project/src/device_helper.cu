#include "device_helper.cuh"

#define INVALID_VAL -1.0f

namespace device_helpers
{
    __device__ void device_helpers::invalidate(cudaSurfaceObject_t grid_map, unsigned int u, unsigned int v)
    {
        unsigned int elem_4_index = u * 16 + 12;
        surf2Dwrite(INVALID_VAL, grid_map, elem_4_index, v);
    }
    
    __device__ bool isInvalid(cudaSurfaceObject_t grid_map, unsigned int u, unsigned int v)
    {
        float validity_indicator;
        unsigned int elem_4_index = u * 16 + 12;
        surf2Dread(&validity_indicator, grid_map, elem_4_index, v);
        
        return validity_indicator == INVALID_VAL;
    }
}
