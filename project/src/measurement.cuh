#include <cuda_runtime.h>
#include "glm_macro.h"
#include <glm/glm.hpp>

#include "cuda_grid_map.h"

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output);
	float downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height);
	float createVertexMap(cudaSurfaceObject_t input_depth, cudaSurfaceObject_t output_vertex, const glm::mat3& inv_cam_k, int width, int height);
	float computeNormalMap(CudaGridMap &depth_map, CudaGridMap &normal_map);

	//Separate these later to more general header file.
	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale);
}

__device__ glm::vec3 computeNormal(cudaSurfaceObject_t  vertex_map, unsigned int u, unsigned int v);