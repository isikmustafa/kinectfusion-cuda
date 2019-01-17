#include "icp.cuh"

#include "device_helper.cuh"

__global__ void constructIcpResidualsKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t target_vertex_map, 
    cudaSurfaceObject_t &target_normal_map, glm::mat3x3 &prev_rot_mat, glm::vec3 &prev_transl_vec, 
    glm::mat3x3 &curr_rot_mat_estimate, glm::vec3 current_transl_vec_estimate, glm::mat3x3 &sensor_intrinsics, 
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

namespace kernel
{
    float constructIcpResiduals(CudaGridMap vertex_map, CudaGridMap target_vertex_map, CudaGridMap target_normal_map, 
        RigidTransform3D & previous_pose, RigidTransform3D current_pose_estimate, glm::mat3x3 & sensor_intrinsics, 
        float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[])
    {
        return 0.0f;
    }
}

__device__ std::array<int, 2> computeCorrespondence(glm::vec3 &vertex_global, glm::mat3x3 &prev_rot_mat, 
    glm::vec3 &prev_transl_vec, glm::mat3x3 &sensor_intrinsics)
{
    // TODO: Implement

	auto point = sensor_intrinsics*glm::inverse(prev_rot_mat)*(vertex_global - prev_transl_vec);

	std::array<int, 2>{ { point.x/point.z, point.y/point.z } };
}

__device__ void writeDummyResidual(float vec_a[], float *scalar_b) 
{
	*scalar_b = 0.0f;
	for (int i = 0; i < 6; i++)
		vec_a[i] = 0.0f;
}

__device__ bool verticesAreTooFarAway(glm::vec3 &vertex_1, glm::vec3 &vertex_2, float distance_thresh) 
{
    return glm::distance(vertex_1, vertex_2) > distance_thresh;
}

//TODO: couldn't find an angle function in glm, should check again
__device__ bool normalsAreTooDifferent(glm::vec3 &normal, glm::vec3 &target_normal, glm::mat3x3 &rotation_mat,
	float angle_thresh) 
{
	glm::vec3 new_normal = normal * rotation_mat;
	glm::vec3 da = glm::normalize(new_normal);
	glm::vec3 db = glm::normalize(target_normal);
	float angle= glm::acos(glm::dot(da, db));
	
    return angle > angle_thresh;
}

__device__ void computeAndFillA(float vec_a[], glm::vec3 &vertex_global, glm::vec3 &target_normal) 
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

__device__ void computeAndFillB(float *scalar_b, glm::vec3 &vertex_global, glm::vec3 &target_vertex,
glm::vec3 &target_normal) 
{
	const auto& s = vertex_global;
	const auto& n = target_normal;
	const auto& d = target_vertex;
	*scalar_b = n.x*d.x + n.y*d.y + n.z*d.z - n.x*s.x - n.y*s.y - n.z*s.z;
}