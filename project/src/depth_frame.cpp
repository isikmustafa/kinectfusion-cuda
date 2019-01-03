#include <iostream>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#include "depth_frame.h"
#include "cuda_utils.h"

DepthFrame::DepthFrame()
{
	cudaChannelFormatDesc half_channel_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);

	//Allocate arrays.
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_raw, &half_channel_desc, 640, 480, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_640x480, &channel_desc, 640, 480, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_320x240, &channel_desc, 320, 240, cudaArraySurfaceLoadStore));
	HANDLE_ERROR(cudaMallocArray(&m_cuda_array_160x120, &channel_desc, 160, 120 , cudaArraySurfaceLoadStore));

	//Create resource descriptors.
	cudaResourceDesc res_desc;
	memset(&res_desc, 0, sizeof(res_desc));
	res_desc.resType = cudaResourceTypeArray;

	//Create CUDA Surface objects
	res_desc.res.array.array = m_cuda_array_raw;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_depth_raw, &res_desc));

	res_desc.res.array.array = m_cuda_array_640x480;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_depth_640x480, &res_desc));

	res_desc.res.array.array = m_cuda_array_320x240;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_depth_320x240, &res_desc));

	res_desc.res.array.array = m_cuda_array_160x120;
	HANDLE_ERROR(cudaCreateSurfaceObject(&m_depth_160x120, &res_desc));
}

DepthFrame::~DepthFrame()
{
	HANDLE_ERROR(cudaDestroySurfaceObject(m_depth_raw));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_depth_640x480));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_depth_320x240));
	HANDLE_ERROR(cudaDestroySurfaceObject(m_depth_160x120));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_raw));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_640x480));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_320x240));
	HANDLE_ERROR(cudaFreeArray(m_cuda_array_160x120));
}

void DepthFrame::update(const std::string& path)
{
	int channel;
	int width;
	int height;
	auto data = stbi_load_16(path.c_str(), &width, &height, &channel, 0);
	if (!data)
	{
		throw std::runtime_error("Error: Image cannot be loaded");
	}

    update(data);
	stbi_image_free(data);
}

void DepthFrame::update(void* data_ptr)
{
	HANDLE_ERROR(cudaMemcpyToArray(m_cuda_array_raw, 0, 0, data_ptr, 640 * 480 * 2, cudaMemcpyHostToDevice));
}

cudaSurfaceObject_t DepthFrame::getRaw() const
{
	return m_depth_raw;
}

std::array<cudaSurfaceObject_t, 3> DepthFrame::getPyramid() const
{
	return { m_depth_640x480, m_depth_320x240, m_depth_160x120 };
}

void DepthFrame::writePyramid() const
{
	writeSurface(m_cuda_array_640x480, 640, 480);
	writeSurface(m_cuda_array_320x240, 320, 240);
	writeSurface(m_cuda_array_160x120, 160, 120);
}

void DepthFrame::writeSurface(cudaArray* gpu_source, int width, int height) const
{
	std::unique_ptr<float[]> float_data(new float[width * height]);
	auto float_data_ptr = float_data.get();

	HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4, cudaMemcpyDeviceToHost));

	std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height]);
	auto byte_data_ptr = byte_data.get();

	int size = width * height;
	for (int i = 0; i < size; ++i)
	{
		byte_data_ptr[i] = static_cast<unsigned char>(float_data_ptr[i] / 200);
	}

	auto final_path = std::to_string(width) + "x" + std::to_string(height) + ".png";
	stbi_write_png(final_path.c_str(), width, height, 1, byte_data_ptr, width);
}