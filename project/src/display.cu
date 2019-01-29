#include "display.cuh"
#include "device_helper.cuh"
#include "glm_macro.h"
#include "cuda_event.h"

#include <cuda_fp16.h>
#include <glm/vec3.hpp>

__global__ void oneHalfChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned short h_pixel;
	surf2Dread(&h_pixel, surface, i * 2, j, cudaBoundaryModeZero);

	auto pixel = static_cast<unsigned char>(__half2float(h_pixel) * scale);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel);

	surf2Dwrite(pixel_w, window, i * 4, j);
}

__global__ void oneFloatChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float f_pixel;
	surf2Dread(&f_pixel, surface, i * 4, j, cudaBoundaryModeZero);

	auto pixel = static_cast<unsigned char>(f_pixel * scale);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel);

	surf2Dwrite(pixel_w, window, i * 4, j);
}

__global__ void fourFloatChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float r, g, b;
	int idx = i * 16;
	surf2Dread(&r, surface, idx, j, cudaBoundaryModeZero);
	surf2Dread(&g, surface, idx + 4, j, cudaBoundaryModeZero);
	surf2Dread(&b, surface, idx + 8, j, cudaBoundaryModeZero);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(b * scale)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(g * scale)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(r * scale));

	surf2Dwrite(pixel_w, window, i * 4, j);
}

__global__ void normalMapToWindowContentKernel(cudaSurfaceObject_t normal_map, cudaSurfaceObject_t window, glm::mat3 inverse_sensor_rotation)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	glm::vec3 normal;
	int idx = i * 16;
	surf2Dread(&normal.x, normal_map, idx, j, cudaBoundaryModeZero);
	surf2Dread(&normal.y, normal_map, idx + 4, j, cudaBoundaryModeZero);
	surf2Dread(&normal.z, normal_map, idx + 8, j, cudaBoundaryModeZero);

	normal = glm::normalize(inverse_sensor_rotation * normal);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(normal.z * 255.0f)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(normal.y * 255.0f)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(normal.x * 255.0f));

	surf2Dwrite(pixel_w, window, i * 4, j);
}

__global__ void shadingToWindowContentKernel(cudaSurfaceObject_t normal_map, cudaSurfaceObject_t window, Sensor sensor)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	glm::vec3 normal;
	int idx = i * 16;
	surf2Dread(&normal.x, normal_map, idx, j, cudaBoundaryModeZero);
	surf2Dread(&normal.y, normal_map, idx + 4, j, cudaBoundaryModeZero);
	surf2Dread(&normal.z, normal_map, idx + 8, j, cudaBoundaryModeZero);

	auto ray_direction = glm::normalize(glm::mat3(sensor.getPose()) * sensor.getInverseIntr(0) * glm::vec3(i + 0.5f, j + 0.5f, 1.0f));
	auto radiance = glm::min(glm::abs(glm::dot(normal, ray_direction)), 1.0f);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(radiance * 255.0f)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(radiance * 255.0f)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(radiance * 255.0f));

	surf2Dwrite(pixel_w, window, i * 4, j);
}

namespace kernel
{
	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		oneHalfChannelToWindowContentKernel << <blocks, threads >> > (surface, window.get_content(), scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		oneFloatChannelToWindowContentKernel << <blocks, threads >> > (surface, window.get_content(), scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, const Window& window, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		fourFloatChannelToWindowContentKernel << <blocks, threads >> > (surface, window.get_content(), scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float normalMapToWindowContent(cudaSurfaceObject_t normal_map, const Window& window, const glm::mat3& inverse_sensor_rotation)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		normalMapToWindowContentKernel << <blocks, threads >> > (normal_map, window.get_content(), inverse_sensor_rotation);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float shadingToWindowContent(cudaSurfaceObject_t normal_map, const Window& window, const Sensor& sensor)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		shadingToWindowContentKernel << <blocks, threads >> > (normal_map, window.get_content(), sensor);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}