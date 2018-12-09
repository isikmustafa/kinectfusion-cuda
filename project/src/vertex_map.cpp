#include "vertex_map.h"
#include "cuda_utils.h"

#include <string>
#include <iostream>
#include <stb_image_write.h>

VertexMap::VertexMap()
{
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	//Allocate arrays.
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_640x480, &channel_desc, 640, 480, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_320x240, &channel_desc, 320, 240, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_160x120, &channel_desc, 160, 120, cudaArraySurfaceLoadStore));

	//Create resource descriptions.
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	//Create CUDA Surface objects
	res_desc.res.array.array = m_cuda_array_640x480;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vertex_640x480, &res_desc));

	res_desc.res.array.array = m_cuda_array_320x240;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vertex_320x240, &res_desc));

	res_desc.res.array.array = m_cuda_array_160x120;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vertex_160x120, &res_desc));
}

VertexMap::~VertexMap()
{
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vertex_640x480));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vertex_320x240));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vertex_160x120));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_640x480));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_320x240));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_160x120));
}

std::array<cudaSurfaceObject_t, 3> VertexMap::getPyramid() const
{
	return { m_vertex_640x480, m_vertex_320x240, m_vertex_160x120 };
}

void VertexMap::writePyramid() const
{
	writeSurface(m_cuda_array_640x480, 640, 480);
	writeSurface(m_cuda_array_320x240, 320, 240);
	writeSurface(m_cuda_array_160x120, 160, 120);
}

void VertexMap::writeSurface(cudaArray* gpu_source, int width, int height) const
{
	std::unique_ptr<float[]> float_data(new float[width * height * 4]);
	auto float_data_ptr = float_data.get();

	HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4 * 4, cudaMemcpyDeviceToHost));

	std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height * 3]);
	auto byte_data_ptr = byte_data.get();

	int size = width * height;
	for (int i = 0; i < size; ++i)
	{
		int idx_byte = i * 3;
		int idx_float = i * 4;
		byte_data_ptr[idx_byte] = static_cast<unsigned char>(float_data_ptr[idx_float] * 0.004f);
		byte_data_ptr[idx_byte + 1] = static_cast<unsigned char>(float_data_ptr[idx_float + 1] * 0.004f);
		byte_data_ptr[idx_byte + 2] = static_cast<unsigned char>(float_data_ptr[idx_float + 2] * 0.004f);
	}

	auto final_path = std::to_string(width) + "x" + std::to_string(height) + "vertex_map.png";
	stbi_write_png(final_path.c_str(), width, height, 3, byte_data_ptr, width * 3);
}