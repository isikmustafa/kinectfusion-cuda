#include "voxel_grid.h"
#include "cuda_utils.h"

VoxelGrid::VoxelGrid(float p_total_width_in_millimeters, int p_n)
	: m_struct(p_total_width_in_millimeters, p_n)
{
	cudaExtent extent = make_cudaExtent(p_n * sizeof(Voxel), p_n, p_n);
	HANDLE_ERROR(cudaMalloc3D(&m_struct.pointer, extent));
	HANDLE_ERROR(cudaMemset3D(m_struct.pointer, 0, extent));
}

VoxelGrid::~VoxelGrid()
{
	HANDLE_ERROR(cudaFree(m_struct.pointer.ptr));
}