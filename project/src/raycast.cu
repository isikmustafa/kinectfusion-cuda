#include "tsdf.cuh"
#include "cuda_event.h"
#include "device_helper.cuh"

__global__ void raycastKernel(VoxelGridStruct voxel_grid, Sensor sensor, cudaSurfaceObject_t output_vertex, cudaSurfaceObject_t output_normal)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	constexpr float min_distance = 0.4f; //Paper suggests this constant.
	auto resolution = voxel_grid.total_width_in_millimeters / voxel_grid.n;
	const float mue = 1.5f * resolution;

	//Parametric ray. ray * d gives the world coordinate position of the d distance from ray's origin.
	glm::vec3 ray = sensor.getPose() * glm::vec4(sensor.getInverseIntr() * glm::vec3(i + 0.5f, j + 0.5f, 1.0f), 1.0f);
	auto distance_increase = mue * 0.99f;
	auto current_distance = min_distance;
	auto current_point = ray * current_distance;

	while (voxel_grid.isPointIn(current_point))
	{
		//1-Check which voxel contains the current_position.
		//2-Check f value of the voxel.
		//3-Update distance_increase based on the f value.
		//After zero crossing
		//4-Determine if it is backfacing or front facing. That is, is it + to - or - to +
		//5-Find surface normal
		//6-Use formula (15) for interpolation.
		//7-Write vertices and normals.
		
		current_distance += distance_increase;
		current_point = ray * current_distance;
	}

}

namespace kernel
{
	float raycast(const VoxelGridStruct& voxel_grid, const Sensor& sensor, cudaSurfaceObject_t output_vertex, cudaSurfaceObject_t output_normal)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		raycastKernel << <blocks, threads >> > (voxel_grid, sensor, output_vertex, output_normal);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}