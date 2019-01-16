#include "icp.cuh"

#include "device_helper.cuh"

__global__ void constructIcpResidualsKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t target_vertex_map, 
    glm::mat3x3 &previous_rotation_mat, glm::vec3 &previous_translation_vec, glm::mat3x3 &sensor_intrinsics, 
    unsigned int width, unsigned int height, float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[])
{
    /* TODO:
        1. Compute indices (u, v) from kernel identifier
        2. Check using device_helpers::is_valid() whether the vertex is valid, else writeDummyResidual() and return
        3. Transform the vertex into the global frame using the previous pose
        4. Run computeCorrespondence() to get the coordinates of the corresponding element of the target vertex map
        5. Check for the distance constraint using verticesAreTooFarAway(), else writeDummyResidual() and return
        6. Compute normals for both, the current vertex map and the predicted vertex map
        7. Check for the angle constraint using normalsAreTooDifferent(), else writeDummyResidual() and return
        8. Compute the parameters for A and write them into the array using computeAndFillA()
        9. Compute the value for b and write it into the array using computeAndFillB()
    */
}

__device__ std::array<int, 2> computeCorrespondence(glm::vec3 &vertex_global, glm::mat3x3 &sensor_intrinsics)
{
    // TODO: Implement
    return std::array<int, 2>();
}
