#pragma once

#include "glm_macro.h"
#include <array>
#include <cuda_runtime.h>
#include <glm/glm.hpp>
#include <cusolverDn.h>

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
	float constructIcpResiduals(CudaGridMap &vertex_map, CudaGridMap &target_vertex_map, CudaGridMap &target_normal_map,
		glm::mat3x3 prev_rot_mat, glm::vec3 prev_transl_vec, glm::mat3x3 curr_rot_mat_estimate,
		glm::vec3 current_transl_vec_estimate, glm::mat3x3 sensor_intrinsics, float distance_thresh, float angle_thresh,
		float *mat_A, float *vec_b);
}

__device__ inline glm::ivec2 computeCorrespondence(glm::vec3 &vertex_global, glm::mat3x3 &prev_rot_mat,
	glm::vec3 &prev_transl_vec, glm::mat3x3 &sensor_intrinsics)
{
	auto point = sensor_intrinsics * glm::transpose(prev_rot_mat) * (vertex_global - prev_transl_vec);
	int u = glm::floor(point.x / point.z);
	int v = glm::floor(point.y / point.z);

	return glm::ivec2(u, v);
}

__device__ inline void writeDummyResidual(float vec_a[], float *scalar_b)
{
	vec_a[0] = 0.0f;
	vec_a[1] = 0.0f;
	vec_a[2] = 0.0f;
	vec_a[3] = 0.0f;
	vec_a[4] = 0.0f;
	vec_a[5] = 0.0f;
	*scalar_b = 0.0f;
}

__device__ inline bool areVerticesTooFarAway(glm::vec3 &vertex_1, glm::vec3 &vertex_2, float distance_threshold)
{
	return glm::distance(vertex_1, vertex_2) > distance_threshold;
}

__device__ inline bool areNormalsTooDifferent(glm::vec3 &normal, glm::vec3 &target_normal, glm::mat3x3 &rotation_mat,
	float angle_threshold)
{
	auto new_normal = glm::normalize(rotation_mat * normal);
	auto angle = glm::acos(glm::dot(new_normal, target_normal));

	return angle > angle_threshold;
}

__device__ inline void computeAndFillA(float vec_a[], glm::vec3 &vertex_global, glm::vec3 &target_normal)
{
	const auto& s = vertex_global;
	const auto& n = target_normal;
	vec_a[0] = n.y*s.z - n.z*s.y;
	vec_a[1] = n.z*s.x - n.x*s.z;
	vec_a[2] = n.x*s.y - n.y*s.x;
	vec_a[3] = n.x;
	vec_a[4] = n.y;
	vec_a[5] = n.z;
}

__device__ inline void computeAndFillB(float *scalar_b, glm::vec3 &vertex_global, glm::vec3 &target_vertex,
	glm::vec3 &target_normal)
{
	const auto& s = vertex_global;
	const auto& n = target_normal;
	const auto& d = target_vertex;

	*scalar_b = n.x*d.x + n.y*d.y + n.z*d.z - n.x*s.x - n.y*s.y - n.z*s.z;
}