#include "cuda_grid_map.h"
#include "depth_map.h"
#include "window.h"
#include "glm_macro.h"

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float convertToDepthMeters(const DepthMap& raw_depth_map, const CudaGridMap& raw_depth_map_meters, float scale);
	float applyBilateralFilter(const CudaGridMap& raw_depth_map_meters, const CudaGridMap& filtered_depth_map);
	float downSample(const CudaGridMap& depth_map, const CudaGridMap& depth_map_downsampled);
	float createVertexMap(const CudaGridMap& depth_map, const CudaGridMap& vertex_map, const glm::mat3& inv_cam_k);
	float computeNormalMap(const CudaGridMap& vertex_map, const CudaGridMap& normal_map);

	//Separate these later to more general header file.
	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
}