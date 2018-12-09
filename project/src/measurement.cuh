#include <cuda_runtime.h>
#include <glm/mat3x3.hpp>

namespace kernel
{
	void applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output);
	void downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height);
	void createVertexMap(cudaSurfaceObject_t input_depth, cudaSurfaceObject_t output_vertex, const glm::mat3& inv_cam_k, int width, int height);
	void createNormalMap(cudaSurfaceObject_t input_vertex, cudaSurfaceObject_t output_normal, int width, int height);
}