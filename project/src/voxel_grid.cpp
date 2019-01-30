#include <vector>
#include <fstream>

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
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.addressMode[1] = cudaAddressModeClamp;
	tex_desc.addressMode[2] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModeLinear;
	tex_desc.normalizedCoords = 1;

	//Create CUDA texture object.
	HANDLE_ERROR(cudaCreateTextureObject(&m_struct.texture_object, &res_desc, &tex_desc, nullptr));
}

VoxelGrid::~VoxelGrid()
{
	HANDLE_ERROR(cudaDestroyTextureObject(m_struct.texture_object));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_struct.surface_object));
	HANDLE_ERROR(cudaFreeArray(m_struct.cuda_array));
}

void VoxelGrid::saveVoxelGrid(std::string file_name)
{
    int n_elements = m_struct.n * m_struct.n * m_struct.n * 2;
    std::vector<float> host_buffer(n_elements);

    size_t n_bytes = n_elements * sizeof(float);
    
    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.srcArray = m_struct.cuda_array;
    copyParams.dstPtr = make_cudaPitchedPtr((void*)&host_buffer[0], m_struct.n * 2 * sizeof(float), 
        m_struct.n, m_struct.n);
    copyParams.extent = make_cudaExtent(m_struct.n, m_struct.n, m_struct.n);;
    copyParams.kind = cudaMemcpyDeviceToHost;

    HANDLE_ERROR(cudaMemcpy3D(&copyParams));
    std::ofstream binary_file(file_name, std::ios::binary);
    binary_file.write((char*)&host_buffer[0], n_bytes);
}

void VoxelGrid::loadVoxelGrid(std::string file_name)
{
    int n_elements = m_struct.n * m_struct.n * m_struct.n * 2;
    size_t n_bytes = n_elements * sizeof(float);

    std::vector<float> host_buffer(n_elements);

    std::ifstream binary_file(file_name, std::ios::binary);
    binary_file.read((char*)&host_buffer[0], n_bytes);

    cudaMemcpy3DParms copyParams = { 0 };
    copyParams.dstArray = m_struct.cuda_array;
    copyParams.srcPtr = make_cudaPitchedPtr((void*)&host_buffer[0], m_struct.n * 2 * sizeof(float), m_struct.n, 
        m_struct.n);
    copyParams.extent = make_cudaExtent(m_struct.n, m_struct.n, m_struct.n);;
    copyParams.kind = cudaMemcpyHostToDevice;

    HANDLE_ERROR(cudaMemcpy3D(&copyParams));
}
