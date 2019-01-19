#include "voxel_grid.h"
#include "cuda_utils.h"

VoxelGrid::VoxelGrid(float p_total_width_in_millimeters, int p_n)
	: m_struct(p_total_width_in_millimeters, p_n)
{
	//Allocate 3D array.
	auto extent = make_cudaExtent(p_n, p_n, p_n);
	auto voxel_channel_desc = cudaCreateChannelDesc(32, 32, 0, 0, cudaChannelFormatKindFloat);
	HANDLE_ERROR(cudaMalloc3DArray(&m_struct.cuda_array, &voxel_channel_desc, extent, cudaArraySurfaceLoadStore));

	//Create resource descriptions.
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;
	res_desc.res.array.array = m_struct.cuda_array;

	//Create CUDA surface object.
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_struct.surface_object, &res_desc));

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(tex_desc));
	tex_desc.addressMode[0] = cudaTextureAddressMode(cudaAddressModeClamp);
	tex_desc.addressMode[1] = cudaTextureAddressMode(cudaAddressModeClamp);
	tex_desc.addressMode[2] = cudaTextureAddressMode(cudaAddressModeClamp);
	tex_desc.filterMode = cudaTextureFilterMode(cudaFilterModeLinear);
	tex_desc.normalizedCoords = true;

	//Create CUDA texture object.
	HANDLE_ERROR(cudaCreateTextureObject(&m_struct.texture_object, &res_desc, &tex_desc, nullptr));
}

VoxelGrid::~VoxelGrid()
{
	HANDLE_ERROR(cudaDestroyTextureObject(m_struct.texture_object));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_struct.surface_object));
	HANDLE_ERROR(cudaFreeArray(m_struct.cuda_array));
}