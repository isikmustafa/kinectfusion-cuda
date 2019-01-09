#include "vector_map.h"

#include <string>
#include <iostream>

#include <stb_image_write.h>
#include "cuda_utils.h"

VectorMap::VectorMap()
{
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	//Allocate arrays.
	HANDLE_ERROR(cudaMallocArray(&m_vectors_640x480.cuda_array, &channel_desc, 640, 480, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_vectors_320x240.cuda_array, &channel_desc, 320, 240, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_vectors_160x120.cuda_array, &channel_desc, 160, 120, cudaArraySurfaceLoadStore));

	//Create resource descriptions.
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	//Create CUDA Surface objects
	res_desc.res.array.array = m_vectors_640x480.cuda_array;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vectors_640x480.surface_object, &res_desc));

    res_desc.res.array.array = m_vectors_320x240.cuda_array;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vectors_320x240.surface_object, &res_desc));

	res_desc.res.array.array = m_vectors_160x120.cuda_array;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_vectors_160x120.surface_object, &res_desc));
}

VectorMap::~VectorMap()
{
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vectors_640x480.surface_object));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vectors_320x240.surface_object));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_vectors_160x120.surface_object));
	HANDLE_ERROR(cudaFreeArray(m_vectors_640x480.cuda_array));
	HANDLE_ERROR(cudaFreeArray(m_vectors_320x240.cuda_array));
	HANDLE_ERROR(cudaFreeArray(m_vectors_160x120.cuda_array));
}

std::array<cudaSurfaceObject_t, 3> VectorMap::getPyramid() const
{
	return { m_vectors_640x480.surface_object, m_vectors_320x240.surface_object, m_vectors_160x120.surface_object};
}

void VectorMap::writePyramidToFile(std::string file_name) const
{
	writeSurface(file_name, m_vectors_640x480.cuda_array, 640, 480);
	writeSurface(file_name, m_vectors_320x240.cuda_array, 320, 240);
	writeSurface(file_name, m_vectors_160x120.cuda_array, 160, 120);
}

void VectorMap::writeSurface(std::string file_name, cudaArray* gpu_source, int width, int height) const
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
		byte_data_ptr[idx_byte] = static_cast<unsigned char>(float_data_ptr[idx_float] * 255.0f);
		byte_data_ptr[idx_byte + 1] = static_cast<unsigned char>(float_data_ptr[idx_float + 1] * 255.0f);
		byte_data_ptr[idx_byte + 2] = static_cast<unsigned char>(float_data_ptr[idx_float + 2] * 255.0f);
	}

	auto final_path = std::to_string(width) + "x" + std::to_string(height) + "normal_map.png";
	stbi_write_png(final_path.c_str(), width, height, 3, byte_data_ptr, width * 3);
}