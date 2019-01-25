#include "window.h"
#include "sensor.h"

#include <cuda_runtime.h>

namespace kernel
{
	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale);
	float normalMapToWindowContent(cudaSurfaceObject_t normal_map, const Window& window, const glm::mat3& inverse_sensor_rotation);
	float shadingToWindowContent(cudaSurfaceObject_t normal_map, const Window& window, const Sensor& sensor);
}