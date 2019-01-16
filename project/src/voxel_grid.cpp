#include "voxel_grid.h"
#include "cuda_utils.h"

VoxelGrid::VoxelGrid(float p_total_width_in_millimeters, int p_n)
	: total_width_in_millimeters(p_total_width_in_millimeters)
	, n(p_n)
{
	cudaExtent extent = make_cudaExtent(n * sizeof(Voxel), n, n);
	HANDLE_ERROR(cudaMalloc3D(&pointer, extent));
	HANDLE_ERROR(cudaMemset3D(pointer, 0, extent));
}

VoxelGrid::~VoxelGrid()
{
	HANDLE_ERROR(cudaFree(pointer.ptr));
}