#include "icp.cuh"

#include <cusolverDn.h>

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
        //dim3 threads(1, 1);
        //dim3 blocks(2, 2);
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




float cudaMatrixMultiplication(float *a, float *b, float * c){
	//matrix - matrix multiplication : c = al * a *b + bet * c

	int m = 6;
	int n = 1;
	int k = 1;

	cudaError_t cudaStat; // cudaMalloc status
	cublasStatus_t stat; // CUBLAS functions status
	cublasHandle_t handle; // CUBLAS context
	int i, j; // i-row index ,j- column index
	 //a mxk matrix a on the host
	 //b kxn matrix b on the host
	 //c mxn matrix c on the host




	a = (float *)malloc(m*k * sizeof(float)); // host memory for a
	b = (float *)malloc(k*n * sizeof(float)); // host memory for b
	c = (float *)malloc(m*n * sizeof(float)); // host memory for c
												// define an mxk matrix a column by column
	
	// on the device
	float * d_a; // d_a - a on the device
	float * d_b; // d_b - b on the device
	float * d_c; // d_c - c on the device
	cudaStat = cudaMalloc((void **)& d_a, m*k * sizeof(*a)); // device
																// memory alloc for a
	cudaStat = cudaMalloc((void **)& d_b, k*n * sizeof(*b)); // device
																// memory alloc for b
	cudaStat = cudaMalloc((void **)& d_c, m*n * sizeof(*c)); // device
																// memory alloc for c
	stat = cublasCreate(&handle); // initialize CUBLAS context
									// copy matrices from the host to the device
	stat = cublasSetMatrix(m, k, sizeof(*a), a, m, d_a, m); //a -> d_a
	stat = cublasSetMatrix(k, n, sizeof(*b), b, k, d_b, k); //b -> d_b
	stat = cublasSetMatrix(m, n, sizeof(*c), c, m, d_c, m); //c -> d_c
	float al = 1.0f; // al =1
	float bet = 1.0f; // bet =1
						// matrix - matrix multiplication : d_c = al*d_a *d_b + bet *d_c
						// d_a -mxk matrix , d_b -kxn matrix , d_c -mxn matrix ;
						// al ,bet -scalars


	stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &al, d_a,
		m, d_b, k, &bet, d_c, m);
	stat = cublasGetMatrix(m, n, sizeof(*c), d_c, m, c, m); // cp d_c - >c
	
	float *result = c;
	cudaFree(d_a); // free device memory
	cudaFree(d_b); // free device memory
	cudaFree(d_c); // free device memory
	cublasDestroy(handle); // destroy CUBLAS context
	free(a); // free host memory
	free(b); // free host memory
	free(c); // free host memory;
	
	return *result;
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
	
	//calculate A*A_T  B*A_T 
	
	
	//
	

	//initialize A and B
	int N = 6;
	double accum; // elapsed time variable
	float *A, *B, *B1; // declare arrays on the host
					   // prepare memory on the host
	A = (float *)malloc(N*N * sizeof(float)); // NxN coeff . matrix
	B = (float *)malloc(N * sizeof(float)); // N- vector rhs B=A*B1
	B1 = (float *)malloc(N * sizeof(float)); // auxiliary N- vect .
	for (int i = 0; i<N*N; i++) A[i] = rand() / (float)RAND_MAX;
	for (int i = 0; i<N; i++) B[i] = 1.0;
	for (int i = 0; i<N; i++) B1[i] = 1.0; // N- vector of ones
	for (int i = 0; i<N; i++) {
		A[i*N + i] = A[i*N + i] + (float)N; // make A positive definite
		for (int j = 0; j<i; j++) A[i*N + j] = A[j*N + i]; // and symmetric
	}
	cublasOperation_t trans = CUBLAS_OP_N;
	cublasStatus_t cublas_status;
	cublasHandle_t handle2;
	cudaError cudaStatus2;
	cudaStatus2 = cudaGetDevice(0);
	cublas_status = cublasCreate_v2(&handle2);
	int value1 = 0;
	cublas_status = cublasGetProperty(MAJOR_VERSION, &value1);
	const float al = 1.0, bet = 0.0; // constants for sgemv
	int incx = 1, incy = 1;
	cublasSgemv_v2(handle2,trans,N,N,&al, /* host or device pointer */
		A,
		N,
		B1,
		incx,
		&bet,  /* host or device pointer */
		B,
		incy);
	
	
	cudaError cudaStatus;
	cusolverStatus_t cusolverStatus;
	cusolverDnHandle_t handle; // device versions of
	float *d_A, *d_B, *Work; // matrix A, rhs B and worksp .
	int *d_info, Lwork; // device version of info , worksp . size
	int info_gpu = 0; // device info copied to host
	cudaStatus = cudaGetDevice(0);
	cusolverStatus = cusolverDnCreate(&handle); // create handle
	cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;
	// prepare memory on the device
	cudaStatus = cudaMalloc((void **)& d_A, N*N * sizeof(float));
	cudaStatus = cudaMalloc((void **)& d_B, N * sizeof(float));
	cudaStatus = cudaMalloc((void **)& d_info, sizeof(int));
	cudaStatus = cudaMemcpy(d_A, A, N*N * sizeof(float),
		cudaMemcpyHostToDevice); // copy A- >d_A
	cudaStatus = cudaMemcpy(d_B, B, N * sizeof(float),
		cudaMemcpyHostToDevice); // copy B- >d_B
								 // compute workspace size and prepare workspace
	cusolverStatus = cusolverDnSpotrf_bufferSize(handle,
		uplo, N, d_A, N, &Lwork);
	cudaStatus = cudaMalloc((void **)& Work, Lwork * sizeof(float));
	//clock_gettime(CLOCK_REALTIME, &start); // start timer
	// Cholesky decomposition d_A =L*L^T, lower triangle of d_A is
	// replaced by the factor L
	cusolverStatus = cusolverDnSpotrf(handle, uplo, N, d_A, N, Work,
		Lwork, d_info);
	// solve d_A *X=d_B , where d_A is factorized by potrf function
	// d_B is overwritten by the solution
	cusolverStatus = cusolverDnSpotrs(handle, uplo, N, 1, d_A, N,
		d_B, N, d_info);
	cudaStatus = cudaDeviceSynchronize();
	//clock_gettime(CLOCK_REALTIME, &stop); // stop timer


	cudaStatus = cudaMemcpy(&info_gpu, d_info, sizeof(int),
		cudaMemcpyDeviceToHost); // copy d_info -> info_gpu
	printf(" after Spotrf + Spotrs : info_gpu = %d\n", info_gpu);
	cudaStatus = cudaMemcpy(B, d_B, N * sizeof(float),
		cudaMemcpyDeviceToHost); // copy solution to host d_B - >B
	printf(" solution : ");
	for (int i = 0; i < 5; i++) printf("%g, ", B[i]); // print
	printf(" ... "); // first components of the solution
	printf("\n");
	// free memory
	cudaStatus = cudaFree(d_A);
	cudaStatus = cudaFree(d_B);
	cudaStatus = cudaFree(d_info);
	cudaStatus = cudaFree(Work);
	/*cusolverStatus = cusolverDnDestroy(handle);
	cublas_status = cublasDestroy_v2(handle2);*/
	cudaStatus = cudaDeviceReset();
	


}