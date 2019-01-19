#include "tsdf.cuh"
#include "cuda_event.h"
#include "device_helper.cuh"

//If there is an intersection between the bbox and ray, parameters (t) of close and far intersection points are returned.
//From https://github.com/isikmustafa/pathtracer/blob/master/bbox.cuh
__device__ glm::vec2 intersectBBox(const glm::vec3& origin, const glm::vec3& inv_dir, const glm::vec3& min, const glm::vec3& max)
{
	auto t0 = (min - origin) * inv_dir;
	auto t1 = (max - origin) * inv_dir;

	auto tmin = fminf(t0.x, t1.x);
	auto tmax = fmaxf(t0.x, t1.x);

	tmin = fmaxf(tmin, fminf(t0.y, t1.y));
	tmax = fminf(tmax, fmaxf(t0.y, t1.y));

	tmin = fmaxf(tmin, fminf(t0.z, t1.z));
	tmax = fminf(tmax, fmaxf(t0.z, t1.z));

	if (tmax < tmin)
	{
		return glm::vec2(-1.0f, -1.0f);
	}

	return glm::vec2(tmin, tmax);
}

__global__ void raycastKernel(VoxelGridStruct voxel_grid, Sensor sensor, cudaSurfaceObject_t output_vertex, cudaSurfaceObject_t output_normal)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	const auto resolution = voxel_grid.resolution;
	const auto mue = voxel_grid.mue;

	auto ray_origin = sensor.getPosition();
	//Do not normalize the direction. pos = origin + dir * depth.
	auto ray_direction = glm::mat3(sensor.getPose()) * sensor.getInverseIntr() * glm::vec3(i + 0.5f, j + 0.5f, 1.0f);

	//For an efficient and correct solution, intersect the ray first with bounding box of the voxel grid to determine near and far distance
	//for ray casting.
	auto half_total_width = voxel_grid.total_width_in_meters * 0.5f;
	auto result = intersectBBox(ray_origin, 1.0f / ray_direction, glm::vec3(-half_total_width), glm::vec3(half_total_width));

	//If voxel grid is behind the camera or has no intersection with this ray.
	if (result.y < 0.0f)
	{
		return;
	}

	auto near_distance = glm::max(kernel::cMinDistance, result.x);
	auto far_distance = glm::min(kernel::cMaxDistance, result.y);

	//If view frustum does not intersect voxel grid from neither sides.
	if (near_distance >= far_distance)
	{
		return;
	}

	auto distance_increase = mue * 0.99f;
	for (auto current_distance = near_distance + resolution * 0.1f; current_distance < far_distance; current_distance += distance_increase)
	{
		auto current_point = ray_origin + ray_direction * current_distance;

		//1-Check which voxel contains the current_position.
		//2-Check f value of the voxel.
		//3-Update distance_increase based on the f value.
		//After zero crossing
		//4-Determine if it is backfacing or front facing. That is, is it + to - or - to +
		//5-Find surface normal
		//6-Use formula (15) for interpolation.
		//7-Write vertices and normals.
	}
}

namespace kernel
{
	float raycast(const VoxelGridStruct& voxel_grid, const Sensor& sensor, const CudaGridMap& output_vertex, const CudaGridMap& output_normal)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		raycastKernel << <blocks, threads >> > (voxel_grid, sensor, output_vertex.getCudaSurfaceObject(), output_normal.getCudaSurfaceObject());
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}