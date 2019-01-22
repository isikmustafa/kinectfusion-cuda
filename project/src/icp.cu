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
    int idx = u * width + v;

    //2. Check whether the vertex is valid
    if (u > width && v > height)
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
	glm::vec2 corresponding_coords = computeCorrespondence(vertex_global, prev_rot_mat, prev_transl_vec, sensor_intrinsics);
    //5. Check for validity of the coordinates 
    if (corresponding_coords.x < 0 && corresponding_coords.y < 0 &&
        corresponding_coords.x >= width && corresponding_coords.y >= height)
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
    glm::vec3 normal = device_helper::computeNormal(vertex_map, u, v);

    //8. Check for the angle constraint
    if (normalsAreTooDifferent(normal, target_normal, curr_rot_mat_estimate, angle_thresh))
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
        std::array<float, 6> mat_A[], float vec_b[])
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
            distance_thresh, angle_thresh, (float (*)[6])&mat_A[0][0], vec_b);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
    }
}




void cudaMatrixMatrixMultiplication(float *mat_left, float *mat_right,
	float *mat_out, int n_rows, cublasOperation_t operation_right)
{
	//matrix - matrix multiplication : c = al * a *b + bet * c

	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context

    stat = cublasCreate(&handle);

	const int m = 6;

	float alpha = 1.0f;
	float beta = 0.0f;

	stat = cublasSgemm(handle, CUBLAS_OP_N, operation_right, m, m, n_rows, &alpha, mat_left,
		m, mat_right, m, &beta, mat_out, m);
	
}
void cudaMatrixVectorMultiplication(float * mat_left, float * vec_right, float *vec_out, int n_rows)
{
	//matrix - matrix multiplication : c = al * a *b + bet * c

	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context

    stat = cublasCreate(&handle);

	const int m = 6;

	float alpha = 1.0f;
	float beta = 1.0f;

	stat = cublasSgemv(handle, CUBLAS_OP_N, m, n_rows, &alpha, mat_left, m, vec_right, 1, &beta,
		vec_out, 1);
}

void solveLinearSystem(float *mat_a, float *vec_b, unsigned int n_equations, float *result_x)
{
    /*
        Variant A: Solve with SVD (probably slowest) as in exercise
        Variant B: Solve with cholesky decomposition, see: http://www.math.iit.edu/~fa
            general instructions (note that in their notation, A* is the transpose, I think)
        More examples and references: 
            https://docs.nvidia.com/cuda/cusolver/index.html
            https://developer.nvidia.com/sites/default/files/akamai/cuda/files/Misc/mygpu.pdf
            https://devtalk.nvidia.com/default/topic/865359/solve-ax-b-with-cusolver/
    */
	// A_t* A= A_t_A A_t* b= A_t_b
	const int n_variables = 6;
	float *A_t_A;
	HANDLE_ERROR(cudaMalloc(&A_t_A, n_equations * n_variables * sizeof(float)));
	
	
	//calculate A*A_T  B*A_T 
	cudaMatrixMatrixMultiplication(mat_a, mat_a, A_t_A, n_equations, CUBLAS_OP_T);
	cudaMatrixVectorMultiplication(mat_a, vec_b, result_x, n_equations);


	cusolverStatus_t cusolverStatus;
	cusolverDnHandle_t handle; // device versions of
	float  *work; // matrix A, rhs B and worksp .
	int *d_info, workspace_size; // device version of info , worksp . size
	int info_gpu = 0; // device info copied to host
	cusolverStatus = cusolverDnCreate(&handle); // create handle
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	// prepare memory on the device
	HANDLE_ERROR(cudaMalloc((void **)& d_info, sizeof(int)));


	cusolverStatus = cusolverDnSpotrf_bufferSize(handle,
		uplo , n_variables, A_t_A, n_variables, &workspace_size);
	HANDLE_ERROR(cudaMalloc((void **)& work, workspace_size * sizeof(float)));
	// Cholesky decomposition d_A =L*L^T, lower triangle of d_A is
	// replaced by the factor L
	cusolverStatus = cusolverDnSpotrf(handle, uplo, n_variables, A_t_A, n_variables, work,
		workspace_size, d_info);
	// solve d_A *X=d_B , where d_A is factorized by potrf function
	// d_B is overwritten by the solution
	cusolverStatus = cusolverDnSpotrs(handle, uplo, n_variables, 1, A_t_A, n_variables,
		result_x, n_variables, d_info);
	HANDLE_ERROR(cudaDeviceSynchronize());


	HANDLE_ERROR(cudaMemcpy(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost)); // copy d_info -> info_gpu
	printf(" after Spotrf + Spotrs : info_gpu = %d\n", info_gpu);
	
	
	// free memory
	HANDLE_ERROR(cudaFree(A_t_A));
	HANDLE_ERROR(cudaFree(d_info));
	HANDLE_ERROR(cudaFree(work));
	


}