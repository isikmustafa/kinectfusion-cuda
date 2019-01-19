#include "icp.cuh"

#include <cusolverDn.h>

#include "device_helper.cuh"
#include "cuda_utils.h"
#include "cuda_event.h"

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

	/* 1. Compute indices (u, v) from thread index*/
	int u = blockIdx.x * blockDim.x + threadIdx.x;
	int v = blockIdx.y * blockDim.y + threadIdx.y;

	int idx = u * 16; //TODO: CHECK IF THIS IS RIGHT
	
	if (u < width && v < height) {
		if (device_helper::isValid(vertex_map, u, v)) { //2. Check whether the vertex is valid
			writeDummyResidual(mat_A[u], &vec_b[v]);
			return;
		}
		glm::vec3 vertex_map_current;
		//surf2Dread(&vertex_map_current, vertex_map, idx, v);
		device_helper::writeVec3(vertex_map_current,vertex_map,u,v);


		glm::vec3 vertex_global = curr_rot_mat_estimate * vertex_map_current+ current_transl_vec_estimate; // 3. Transform the vertex into the global frame
		std::array<int, 2> cor_point = computeCorrespondence(vertex_global, prev_rot_mat, prev_transl_vec, sensor_intrinsics); // 4. Run computeCorrespondence()

		/*if (cor_point[0] >= 0 && cor_point[1] >= 0 && cor_point[0] < width && cor_point[1] < height) //5. Check for validity of the coordinates 
		{
			glm::vec3 vertex_map_target;
			//surf2Dread(&vertex_map_target, target_vertex_map, idx+4, v);//TODO: CHECK
			if(verticesAreTooFarAway(vertex_map_current, vertex_map_target, distance_thresh) ){ // 6. Check for the distance constraint
				writeDummyResidual(mat_A[u], &vec_b[v]);
				return;
			}
			else {
				glm::vec3 target_normal;
				//surf2Dread(&target_normal, target_normal_map, idx+8, v); //TODO: CHECK
				glm::vec3 normal = device_helper::computeNormal(vertex_map, u, v); //7. Compute the normal for the vertex
				if (normalsAreTooDifferent(normal, target_normal, curr_rot_mat_estimate, angle_thresh))//8. Check for the angle constraint
				{
					writeDummyResidual(mat_A[u], &vec_b[v]);
					return;
				}
				else {
					computeAndFillA(mat_A[u], vertex_global, target_normal);//9. Compute the parameters for A 
					computeAndFillB(&vec_b[v], vertex_global, vertex_map_target, target_normal);//10. Compute the parameters for B 
				}
				
			}

			

		}*/

	}
	


	
}

namespace kernel
{
    float constructIcpResiduals(CudaGridMap vertex_map, CudaGridMap target_vertex_map, CudaGridMap target_normal_map, 
		glm::mat3x3 &prev_rot_mat, glm::vec3 &prev_transl_vec,
		glm::mat3x3 &curr_rot_mat_estimate, glm::vec3 current_transl_vec_estimate, glm::mat3x3 & sensor_intrinsics,
        float distance_thresh, float angle_thresh, float mat_A[][6], float vec_b[])
    {
		auto dims = vertex_map.getGridDims();

		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(dims[0] / threads.x, dims[1] / threads.y);
		start.record();
		/*constructIcpResidualsKernel << <blocks, threads >> > (vertex_map.getCudaSurfaceObject(),  target_vertex_map.getCudaSurfaceObject(),
			target_normal_map.getCudaSurfaceObject(),  prev_rot_mat,  prev_transl_vec,
		    curr_rot_mat_estimate, current_transl_vec_estimate,  sensor_intrinsics,
			dims[0], dims[1], distance_thresh,  angle_thresh,  mat_A[][6], vec_b[]);*/
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
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