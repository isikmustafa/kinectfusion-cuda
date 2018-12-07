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

	if (i % 2 == 1 || j % 2 == 1)
	{
		return;
	}

	//Just average for now. Do the same as in the paper later on.
	int idx = i * 4;
	float f1, f2, f3, f4;
	surf2Dread(&f1, source, idx, j);
	surf2Dread(&f2, source, idx + 4, j);
	surf2Dread(&f3, source, idx, j + 1);
	surf2Dread(&f4, source, idx + 4, j + 1);

	surf2Dwrite((f1 + f2 + f3 + f4) * 0.25f, destination, idx / 2, j / 2);
}