#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void applyBilateralFilter(cudaSurfaceObject_t raw, cudaSurfaceObject_t filtered)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned short v1;
	surf2Dread(&v1, raw, i * 2, j);

	surf2Dwrite(__half2float(v1), filtered, i * 4, j);
}

__global__ void downSample(cudaSurfaceObject_t source, cudaSurfaceObject_t destination)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	//Just average for now. Do the same as in the paper later on.
	int idx_i = i * 8;
	int idx_j = j * 2;
	float f1, f2, f3, f4;
	surf2Dread(&f1, source, idx_i, idx_j);
	surf2Dread(&f2, source, idx_i + 4, idx_j);
	surf2Dread(&f3, source, idx_i, idx_j + 1);
	surf2Dread(&f4, source, idx_i + 4, idx_j + 1);

	surf2Dwrite((f1 + f2 + f3 + f4) * 0.25f, destination, i * 4, j);
}