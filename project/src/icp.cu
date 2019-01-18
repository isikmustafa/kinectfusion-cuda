#include "icp.cuh"

#include <cusolverDn.h>

#include "device_helper.cuh"
#include "cuda_utils.h"

__global__ void constructIcpResidualsKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t target_vertex_map, 
    cudaSurfaceObject_t target_normal_map, glm::mat3x3 &prev_rot_mat, glm::vec3 &prev_transl_vec, 
    glm::mat3x3 &curr_rot_mat_estimate, glm::vec3 current_transl_vec_estimate, glm::mat3x3 &sensor_intrinsics, 
    unsigned int width, unsigned int height, float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[])
{
    /* TODO:
        1. Compute indices (u, v) from thread index
        2. Check using device_helper::is_valid() whether the vertex is valid, else writeDummyResidual() and return
        3. Transform the vertex into the global frame using the current pose estimate
        4. Run computeCorrespondence() to get the coordinates of the corresponding element of the target vertex map
        5. Check for validity of the coordinates (not negative, not larger or equal than height/width)
        6. Check for the distance constraint using verticesAreTooFarAway(), else writeDummyResidual() and return
        7. Compute the normal for the vertex (in global frame) using computeNormal() from measurement.cuh
        8. Check for the angle constraint using normalsAreTooDifferent(), else writeDummyResidual() and return
        9. Compute the parameters for A and write them into the array using computeAndFillA()
        10. Compute the value for b and write it into the array using computeAndFillB()
    */
}

namespace kernel
{
    float constructIcpResiduals(CudaGridMap vertex_map, CudaGridMap target_vertex_map, CudaGridMap target_normal_map, 
        RigidTransform3D & previous_pose, RigidTransform3D current_pose_estimate, glm::mat3x3 & sensor_intrinsics, 
        float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[])
    {
        return 0.0f;
    }
}

void solveLinearSystem(std::array<float, 6> *mat_a, float *vec_b, unsigned int n_equations,
    std::array<float, 6> *result_x)
{
    /*
        Variant A: Solve with SVD (probably slowest) as in exercise
        Variant B: Solve with cholesky decomposition, see: http://www.math.iit.edu/~fa
            general instructions (note that in their notation, A* is the transpose, I thi
        More examples and references: 
            https://docs.nvidia.com/cuda/cusolver/index.html
            https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
            https://devtalk.nvidia.com/default/topic/865359/solve-ax-b-with-cusolver/
    */
}