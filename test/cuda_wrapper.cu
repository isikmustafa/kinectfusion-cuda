#include "cuda_wrapper.cuh"

__global__ void computeCorrespondenceTestKernel(std::array<int, 2> *result_coords, cudaSurfaceObject_t vertices, 
    glm::mat3x3 &sensor_intrinsics)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    glm::vec3 vertex_global;
    int idx = i * 16;
    surf2Dread(&(vertex_global.x), vertices, idx, j);
    surf2Dread(&(vertex_global.y), vertices, idx + 4, j);
    surf2Dread(&(vertex_global.z), vertices, idx + 8, j);
    result_coords[i * 2 + j] = computeCorrespondence(vertex_global, sensor_intrinsics);
}


void computeCorrespondenceTestWrapper(std::array<int, 2> *result, CudaGridMap &vertex_map, glm::mat3x3 &intrinsics)
{
    computeCorrespondenceTestKernel<<<1,4>>>(result, vertex_map.getCudaSurfaceObject(), intrinsics);
}