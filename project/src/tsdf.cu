#include "tsdf.cuh"
#include "cuda_event.h"
#include "device_helper.cuh"

#include <cuda_fp16.h>

__global__ void fuseKernel(cudaSurfaceObject_t raw_depth_map, VoxelGridStruct voxel_grid, Sensor sensor)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	auto* char_ptr = static_cast<char*>(voxel_grid.pointer.ptr);
	auto* ptr = reinterpret_cast<Voxel*>(char_ptr + voxel_grid.pointer.pitch * (voxel_grid.n * i + j));

	const auto resolution = voxel_grid.resolution;
	const auto mue = voxel_grid.mue;
	auto start_coord = (-voxel_grid.total_width_in_meters + resolution) * 0.5f;
	glm::vec3 point(start_coord + i * resolution, start_coord + j * resolution, start_coord - resolution);
	for (int k = 0; k < voxel_grid.n; ++k)
	{
		//Update the point to the center of the next voxel.
		point.z += resolution;

		//1-Project point into image space of sensor and perform homogeneous division.
		glm::vec3 point_eye = sensor.getInversePose() * glm::vec4(point, 1.0f);
		auto pixel = sensor.getIntr() * point_eye;
		pixel /= pixel.z;

		//2-Check if it is in the view frustum. If not, don't do anything.
		if (!(point_eye.z >= kernel::cMinDistance && point_eye.z < kernel::cMaxDistance &&
			pixel.x >= 0.0f && pixel.x < 640.0f && pixel.y >= 0.0f && pixel.y < 480.0f))
		{
			continue;
		}

		//3-Take depth value by nearest neighbour lookup.
		unsigned short h_depth;
		surf2Dread(&h_depth, raw_depth_map, static_cast<int>(pixel.x) * 2, static_cast<int>(pixel.y));
		auto depth = __half2float(h_depth);

		//If depth value is invalid, continue with the next voxel.
		if (!device_helper::isDepthValid(depth))
		{
			continue;
		}

		auto diff = depth - point_eye.z;
		if (diff >= -mue)
		{
			//4-Compute TSDF and weight;
			auto f = glm::min(1.0f, diff / mue) * glm::sign(diff);
			auto w = 1.0f;

			//5-Update voxel.f and voxel.w
			auto& voxel = ptr[k];
			voxel.f = (voxel.f * voxel.w + f * w) / (voxel.w + w);
			voxel.w += w;
		}
	}
}

namespace kernel
{
	float fuse(cudaSurfaceObject_t raw_depth_map, const VoxelGridStruct& voxel_grid, const Sensor& sensor)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(voxel_grid.n / threads.x, voxel_grid.n / threads.y);
		start.record();
		fuseKernel <<<blocks, threads>>> (raw_depth_map, voxel_grid, sensor);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}