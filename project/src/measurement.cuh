#include "cuda_grid_map.h"
#include "depth_map.h"
#include "window.h"
#include "glm_macro.h"

#include <cuda_runtime.h>
#include <glm/glm.hpp>

namespace kernel
{
	//All kernel wrapper functions returns the time it takes to execute them in milliseconds.
	float convertToDepthMeters(const DepthMap& raw_depth_map, const CudaGridMap& raw_depth_map_meters, float scale, bool unmirror);
	float applyBilateralFilter(const CudaGridMap& raw_depth_map_meters, const CudaGridMap& filtered_depth_map);
	float downSample(const CudaGridMap& depth_map, const CudaGridMap& depth_map_downsampled);
	float createVertexMap(const CudaGridMap& depth_map, const CudaGridMap& vertex_map, const glm::mat3& inv_cam_k);
	float createNormalMap(const CudaGridMap& vertex_map, const CudaGridMap& normal_map);
}