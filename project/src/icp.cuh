#include <array>
#include <cuda_runtime.h>
#include "glm_macro.h"
#include <glm/glm.hpp>

#include "cuda_grid_map.h"
#include "rigid_transform_3d.h"

namespace kernel
{
    /*
    Kernel wrapper for a single ICP iteration, operating on a single resolution of vertex map. The kernel fills the
    matrix A and the vector b with the parameters for the linear system of equations of the form: A * x = b.
    Note that here A is the full matrix, in Ismar et al. (2011) A corresponds only to one row.
    Every kernel operates on only one vertex and fills one row of A (and the corresponding element of b). It is
    expected that there is enough space allocated for A and b. For invalid vertices or if no correspondence that
    fulfills all constraints can be found for a vertex, all elements of the corresponding row of A and the element
    of b are set to 0, which represents a dummy residual.
    Parameters:
        - vertex_map: grid map containing the vertices for the current measurement (time k) --> V_k(u)
        - target_vertex_map: grid map the vertices that were predicted from the current TSDF model represented in global
            frame --> V_{g, k-1}(u^)
        - previous_pose: rigid transformation object containing the pose of the previous frame --> R_{g, k-1}
        - current_pose_estimate: rigid transformation object containing the current estimated camera pose 
            --> ~R_{g, k}^y
        - distance_thresh: float, specifying the maximum distance between corresponding vectors
        - angle_thresh: float, specifying the maximum angle between corrsponding normals
        - mat_A: 2D float array of size (width * height) x 6, representing the matrix A
        - vec_b: 1D float array of length (width * height) representing the vector b
*/
    float constructIcpResiduals(CudaGridMap vertex_map, CudaGridMap target_vertex_map, CudaGridMap target_normal_map,
        RigidTransform3D &previous_pose, RigidTransform3D current_pose_estimate, glm::mat3x3 &sensor_intrinsics,
        float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[]);
}

// Function that fills the given vector and the given scalar with zeroes.
__device__ void writeDummyResidual(float vec_a[], float *scalar_b);

// Function that performs projective data association and returns the coordinates of the correspondence.
__device__ std::array<int, 2> computeCorrespondence(glm::vec3 &vertex_global, glm::mat3x3 &prev_rot_mat, 
    glm::vec3 &prev_transl_vec, glm::mat3x3 &sensor_intrinsics);

// Returns true if the distance between the vertices is above the specified threshold. Both vertices are expected in the
// same frame
__device__ bool verticesAreTooFarAway(glm::vec3 &vertex_1, glm::vec3 &vertex_2, float distance_thresh);

// Returns true if the angle between the normals is above the specified threshold. It is expected that the rotation
// matrix transforms the first normal into the frame of the 2nd normal
__device__ bool normalsAreTooDifferent(glm::vec3 &normal, glm::vec3 &target_normal, glm::mat3x3 &rotation_mat,
    float angle_thresh);

// The function computes the 6 parameters that make one row of A and write them into the array.
__device__ void computeAndFillA(float vec_a[], glm::vec3 &vertex_global, glm::vec3 &target_normal);

// The function computes the value for b and writes it into the array.
__device__ void computeAndFillB(float *scalar_b, glm::vec3 &vertex_global, glm::vec3 &target_vertex, 
    glm::vec3 &target_normal);

void solveLinearSystem(std::array<float, 6> *mat_a, float *vec_b, unsigned int n_equations,
    std::array<float, 6> *result_x);