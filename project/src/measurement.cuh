#include <cuda_runtime.h>

namespace kernel
{
	void applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output);
	void downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height);
}