#include "tsdf.cuh"
#include "cuda_event.h"

__global__ void fuseKernel(cudaSurfaceObject_t raw_depth_map, VoxelGrid voxel_grid, Sensor sensor)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	auto* char_ptr = static_cast<char*>(voxel_grid.pointer.ptr);
	auto* ptr = reinterpret_cast<Voxel*>(char_ptr + voxel_grid.pointer.pitch * (voxel_grid.n * i + j));

	auto resolution = voxel_grid.total_width_in_millimeters / voxel_grid.n;
	auto start_coord = (-voxel_grid.total_width_in_millimeters + resolution) * 0.5f;
	glm::vec3 point(start_coord + i * resolution, start_coord + j * resolution, start_coord);
	for (int k = 0; k < voxel_grid.n; ++k)
	{
		auto& voxel = ptr[k];
		//1-Project point into image space of sensor.
		//2-Check if it is in the view frustum. x >= 0.0 && x <= 640 && y >=0 && y <= 480. If not, don't do anything.
		//3-Take depth value by nearest neighbour lookup.
		//--Decide on truncation value, mue.
		//4-Compute depth difference and compute TSDF.
		//5-Make weight = 1 for this time. Then, change it to what paper does.
		//6-Update voxel.f and voxel.w

		//Update the point to the center of the next voxel.
		point.z += resolution;
	}
}

namespace kernel
{
	float fuse(cudaSurfaceObject_t raw_depth_map, const VoxelGrid& voxel_grid, const Sensor& sensor)
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