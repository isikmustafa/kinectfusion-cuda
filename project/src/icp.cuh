#include <array>
#include <cuda_runtime.h>
#include "glm_macro.h"
#include <glm/glm.hpp>

/*
    Main kernel for a single ICP iteration, operating on a single resolution of vertex map. The kernel fills the 
    matrix A and the vector b with the parameters for the linear system of equations of the form: A * x = b.
    Note that here A is the full matrix, in Ismar et al. (2011) A corresponds only to one row.
    Every kernel operates on only one vertex and fills one row of A (and the corresponding element of b). It is 
    expected that there is enough space allocated for A and b. For invalid vertices or if no correspondence that
    fulfills all constraints can be found for a vertex, all elements of the corresponding row of A and the element 
    of b are set to 0, which represents a dummy residual.
    Parameters:
        - vertex_map: cuda surface object containing the vertices for the current measurement (time k) --> V_k(u)
        - target_vertex_map: surface object containing the vertices that were predicted from the current TSDF model
            represented in global frame --> V_{g, k-1}(u^)
        - previous_rotation_mat: 3x3 glm matrix containing the rotation matrix for the previous frame --> R_{g, k-1}
        - previous_translation_vec: 3D glm vector containing the translation vector for the previous frame 
            --> t_{g, k-1}
        - width: integer giving the width of the vertex map
        - height: integer giving the height of the vertex map
        - mat_A: 2D float array of size (width * height) x 6, representing the matrix A
        - vec_b: 1D float array of length (width * height) representing the vector b
*/
__global__ void constructIcpResidualsKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t target_vertex_map, 
    glm::mat3x3 &previous_rotation_mat, glm::vec3 &previous_translation_vec, glm::mat3x3 &sensor_intrinsics,
    unsigned int width, unsigned int height, float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[]);

// Function that fills the given vector and the given scalar with zeroes.
__device__ void writeDummyResidual(float vec_a[], float *scalar_b);

// Function that performs projective data association and returns the coordinates of the correspondence.
__device__ std::array<int, 2> computeCorrespondence(glm::vec3 &vertex_global, glm::mat3x3 &sensor_intrinsics);

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