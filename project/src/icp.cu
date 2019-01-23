#include "icp.cuh"

#include "device_helper.cuh"
#include "cuda_utils.h"
#include "cuda_event.h"

#define IDX2C(i,j,ld) (((j)*(ld))+(i))


__global__ void constructIcpResidualsKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t target_vertex_map, 
    cudaSurfaceObject_t target_normal_map, glm::mat3x3 prev_rot_mat, glm::vec3 prev_transl_vec, 
    glm::mat3x3 curr_rot_mat_estimate, glm::vec3 current_transl_vec_estimate, glm::mat3x3 sensor_intrinsics, 
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
	/* 1. Compute indices (u, v) from thread index*/
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = v * width + u;

    //2. Check whether the vertex is valid
    if (u > width || v > height)
    {
        return;
    }
	if (!device_helper::isValid(vertex_map, u, v))
    {
		writeDummyResidual(mat_A[idx], &vec_b[idx]);
		return;
	}

    // 3. Transform the vertex into the global frame
    glm::vec3 vertex_camera = device_helper::readVec3(vertex_map, u, v);
	glm::vec3 vertex_global = curr_rot_mat_estimate * vertex_camera + current_transl_vec_estimate; 

    // 4. Run computeCorrespondence()
	glm::ivec2 corresponding_coords = computeCorrespondence(vertex_global, prev_rot_mat, prev_transl_vec, 
        sensor_intrinsics);
    int u_corr = corresponding_coords.x;
    int v_corr = corresponding_coords.y;

    //5. Check for validity of the correspondence
    if (u_corr < 0 || v_corr < 0 || u_corr >= width || v_corr >= height ||
        !device_helper::isValid(target_vertex_map, u_corr, v_corr) ||
        !device_helper::isValid(target_normal_map, u_corr, v_corr))
    {
        writeDummyResidual(mat_A[idx], &vec_b[idx]);
        return;
    }

	glm::vec3 target_vertex = device_helper::readVec3(target_vertex_map, u, v);
    
    // 6. Check for the distance constraint
	if(verticesAreTooFarAway(vertex_global, target_vertex, distance_thresh) )
	{ 
	    writeDummyResidual(mat_A[idx], &vec_b[idx]);
	    return;
	}

    glm::vec3 target_normal = device_helper::readVec3(target_normal_map, u, v);

    //7. Compute the normal for the vertex
    glm::vec4 normal = device_helper::computeNormal(vertex_map, u, v);
    if (normal.w == device_helper::cInvalid)
    {
        writeDummyResidual(mat_A[idx], &vec_b[idx]);
        return;
    }

    //8. Check for the angle constraint
    if (normalsAreTooDifferent(glm::vec3(normal), target_normal, curr_rot_mat_estimate, angle_thresh))
    {
        writeDummyResidual(mat_A[idx], &vec_b[idx]);
        return;
    }

    //9. Compute the parameters for A 
    computeAndFillA(mat_A[idx], vertex_global, target_normal);

    //10. Compute the parameters for B 
    computeAndFillB(&vec_b[idx], vertex_global, target_vertex, target_normal);
}

namespace kernel
{
    float constructIcpResiduals(CudaGridMap &vertex_map, CudaGridMap &target_vertex_map, CudaGridMap &target_normal_map, 
		glm::mat3x3 prev_rot_mat, glm::vec3 prev_transl_vec, glm::mat3x3 curr_rot_mat_estimate, 
        glm::vec3 current_transl_vec_estimate, glm::mat3x3 sensor_intrinsics, float distance_thresh, float angle_thresh, 
        float *mat_A, float *vec_b)
    {
		auto dims = vertex_map.getGridDims();

		CudaEvent start, end;
        int temp = std::min<int>(dims[0] - 1, 8);
        dim3 threads(std::min<int>(dims[0] - 1, 8), std::min<int>(dims[1] - 1, 8));
		dim3 blocks(dims[0] / threads.x, dims[1] / threads.y);
        
		start.record();
		constructIcpResidualsKernel<<<blocks, threads>>>(vertex_map.getCudaSurfaceObject(),  
            target_vertex_map.getCudaSurfaceObject(), target_normal_map.getCudaSurfaceObject(),  prev_rot_mat,  
            prev_transl_vec, curr_rot_mat_estimate, current_transl_vec_estimate, sensor_intrinsics, dims[0], dims[1], 
            distance_thresh, angle_thresh, (float(*)[6])mat_A, vec_b);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
    }
}