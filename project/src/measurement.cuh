#include <cuda_runtime.h>
#include <glm/mat3x3.hpp>

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output);
	float downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height);
	float createVertexMap(cudaSurfaceObject_t input_depth, cudaSurfaceObject_t output_vertex, const glm::mat3& inv_cam_k, int width, int height);
	float createNormalMap(cudaSurfaceObject_t input_vertex, cudaSurfaceObject_t output_normal, int width, int height);

	//Separate these later to more general header file.
	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
}