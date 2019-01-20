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

__device__ glm::vec3 computeGradient(const glm::vec3& point, const VoxelGridStruct& voxel_grid)
{
	auto uvw_resolution = 1.0f / voxel_grid.n;
	auto uvw = point / voxel_grid.total_width_in_meters + glm::vec3(0.5f);

	auto f = tex3D<float2>(voxel_grid.texture_object, uvw.x, uvw.y, uvw.z).x;
	auto f_x = tex3D<float2>(voxel_grid.texture_object, uvw.x - uvw_resolution, uvw.y, uvw.z).x;
	auto f_y = tex3D<float2>(voxel_grid.texture_object, uvw.x, uvw.y - uvw_resolution, uvw.z).x;
	auto f_z = tex3D<float2>(voxel_grid.texture_object, uvw.x, uvw.y, uvw.z - uvw_resolution).x;

	return (glm::vec3(f_x, f_y, f_z) - glm::vec3(f)) / uvw_resolution;
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
	auto previous_tsdf = 0.0f;
	auto precise_distance = 0.0f;
	for (auto current_distance = near_distance; current_distance < far_distance; current_distance += distance_increase)
	{
		//1-Find the current point on the ray.
		auto current_point = ray_origin + ray_direction * current_distance;

		//2-Find trilinearly interpolated TSDF value of the current_point.
		auto uvw = current_point / voxel_grid.total_width_in_meters + glm::vec3(0.5f);
		auto tsdf = tex3D<float2>(voxel_grid.texture_object, uvw.x, uvw.y, uvw.z).x;

		//3-Check TSDF value of the voxel and determine if this is a zero crossing or not.
		if (tsdf < 0.0f)
		{
			//If this is not the first iteration, it means the ray does not intersect a backfacing surface
			//and continued from +ve to -ve.
			if (current_distance != near_distance)
			{
				//Formula(15) to compute more precise distance of intersection.
				precise_distance = current_distance - distance_increase * previous_tsdf / (tsdf - previous_tsdf);
			}
			break;
		}
		//4-Update distance_increase if it is the region of uncertainty.
		else if (tsdf < 0.99f)
		{
			distance_increase = mue * 0.125f;
		}
		previous_tsdf = tsdf;
	}

	//Ray intersected a surface and no backfacing surface is found.
	if (precise_distance != 0.0f)
	{
		auto vertex = ray_origin + ray_direction * precise_distance;
		auto normal = computeGradient(vertex, voxel_grid);

		//Convert normal and the vertex back to eye space.
		//Because the normals and vertices in measurement stage are in eye space.
		vertex = sensor.getInversePose() * glm::vec4(vertex, 1.0f);
		normal = glm::mat3(sensor.getInversePose()) * normal;
		auto normal_norm = glm::length(normal);

		if (device_helper::isDepthValid(normal_norm))
		{
			//Write vertex.
			device_helper::writeVec3(vertex, output_vertex, i, j);
			device_helper::validate(output_vertex, i, j);

			//Write normal.
			normal /= normal_norm;
			device_helper::writeVec3(normal, output_normal, i, j);

			return;
		}
	}

	device_helper::writeVec3(glm::vec3(0.0f), output_vertex, i, j);
	device_helper::invalidate(output_vertex, i, j);
	device_helper::writeVec3(glm::vec3(0.0f), output_normal, i, j);
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