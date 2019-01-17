#include "cuda_wrapper.cuh"

__global__ void computeCorrespondenceTestKernel(std::array<int, 2> *result_coords, cudaSurfaceObject_t vertices, 
    glm::mat3x3 &rotation_mat, glm::vec3 &translation_vec, glm::mat3x3 &sensor_intrinsics)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    
    glm::vec3 vertex_global;
    int idx = i * 16;
    surf2Dread(&(vertex_global.x), vertices, idx, j);
    surf2Dread(&(vertex_global.y), vertices, idx + 4, j);
    surf2Dread(&(vertex_global.z), vertices, idx + 8, j);
    result_coords[i * 2 + j] = computeCorrespondence(vertex_global, rotation_mat, translation_vec, sensor_intrinsics);
}

void computeCorrespondenceTestWrapper(std::array<int, 2> *result, CudaGridMap &vertex_map, glm::mat3x3 &rotation_mat,
    glm::vec3 &translation_vec, glm::mat3x3 &intrinsics)
{
    cudaSurfaceObject_t vertices = vertex_map.getCudaSurfaceObject();
    computeCorrespondenceTestKernel<<<1, dim3(2, 2)>>>(result, vertices, rotation_mat, translation_vec, intrinsics);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void cumputeNormalTestKernel(glm::vec3 *normal, cudaSurfaceObject_t vertices, unsigned int u, unsigned int v)
{
    *normal = computeNormal(vertices, u, v);
}

glm::vec3 computeNormalTestWrapper(CudaGridMap &vertex_map, unsigned int u, unsigned int v)
{
    glm::vec3 *normal_device;
    int size = sizeof(glm::vec3);
    HANDLE_ERROR(cudaMalloc(&normal_device, size));

    cumputeNormalTestKernel<<<1, 1>>>(normal_device, vertex_map.getCudaSurfaceObject(), u, v);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    glm::vec3 normal_host;
    HANDLE_ERROR(cudaMemcpy(&normal_host, normal_device, size, cudaMemcpyDeviceToHost));
    
    return normal_host;

}


__global__ void normalsAreTooDifferentTestKernel(bool *result, glm::vec3 normal, glm::vec3 target_normal, 
    glm::mat3x3 rotation_mat, float angle_thresh)
{
    *result = normalsAreTooDifferent(normal, target_normal, rotation_mat, angle_thresh);
}

bool normalsAreTooDifferentTestWrapper(glm::vec3 normal, glm::vec3 target_normal, glm::mat3x3 rotation_mat, 
    float angle_thresh)
{
    bool *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(bool)));

    normalsAreTooDifferentTestKernel<<<1, 1>>>(result_device, normal, target_normal, rotation_mat, angle_thresh);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    bool result_host;
    HANDLE_ERROR(cudaMemcpy(&result_host, result_device, sizeof(bool), cudaMemcpyDeviceToHost));

    return result_host;
}


__global__ void computeAndFillATestKernel(float *mat_a, glm::vec3 vertex, glm::vec3 normal)
{
    computeAndFillA(mat_a, vertex, normal);
}

void computeAndFillATestWrapper(std::array<float, 6> *result, glm::vec3 vertex, glm::vec3 normal)
{
    float *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(float) * 6));

    computeAndFillATestKernel<<<1, 1>>>(result_device, vertex, normal);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaMemcpy(result, result_device, sizeof(float) * 6, cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaDeviceSynchronize());
}


__global__ void computeAndFillBTestKernel(float *scalar_b, glm::vec3 vertex, glm::vec3 target_vertex, 
    glm::vec3 target_normal)
{
    computeAndFillB(scalar_b, vertex, target_vertex, target_normal);
}

float computeAndFillBTestWrapper(glm::vec3 vertex, glm::vec3 target_vertex, glm::vec3 target_normal)
{
    float *result_device;
    HANDLE_ERROR(cudaMalloc(&result_device, sizeof(float)));

    computeAndFillBTestKernel<<<1, 1>>> (result_device, vertex, target_vertex, target_normal);
    HANDLE_ERROR(cudaPeekAtLastError());
    HANDLE_ERROR(cudaDeviceSynchronize());

    float result_host;
    HANDLE_ERROR(cudaMemcpy(&result_host, result_device, sizeof(float), cudaMemcpyDeviceToHost));
    return result_host;
}
