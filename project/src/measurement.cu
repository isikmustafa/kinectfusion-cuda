#define GLM_FORCE_PURE
#include "measurement.cuh"

#include <iostream>

#include <cuda_fp16.h>
#include <glm/vec3.hpp>
#include "cuda_event.h"

//TODO: How do we decide?
constexpr float cSigmaS = 4.0f;
constexpr float cSigmaR = 0.25f;

__global__ void applyBilateralFilterKernel(cudaSurfaceObject_t raw, cudaSurfaceObject_t filtered)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	constexpr int w_size = 7;

	//Do not change.
	constexpr int half_w_size = w_size / 2;
	constexpr float one_over_sigmasqr_s = 1.0f / (cSigmaS * cSigmaS);
	constexpr float one_over_sigmasqr_r = 1.0f / (cSigmaR * cSigmaR);

	unsigned short h_center, h_current;
	surf2Dread(&h_center, raw, i * 2, j);
	auto center = __half2float(h_center);
	auto normalization = 0.0f;
	auto acc = 0.0f;
	for (int x = -half_w_size; x <= half_w_size; ++x)
	{
		for (int y = -half_w_size; y <= half_w_size; ++y)
		{
			surf2Dread(&h_current, raw, (i + x) * 2, j + y, cudaBoundaryModeClamp);
			auto current = __half2float(h_current);

			auto s_dist_sqr = static_cast<float>(x * x + y * y);
			auto i_dist_sqr = (center - current);
			i_dist_sqr *= i_dist_sqr;
			auto factor = expf(-s_dist_sqr * one_over_sigmasqr_s - i_dist_sqr * one_over_sigmasqr_r);
			normalization += factor;

			acc += factor * current;
		}
	}

	surf2Dwrite(acc / normalization, filtered, i * 4, j);
}

__global__ void downSampleKernel(cudaSurfaceObject_t source, cudaSurfaceObject_t destination)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	//Just average for now. Do the same as in the paper later on.
	int idx_i = i * 8;
	int idx_j = j * 2;
	float f1, f2, f3, f4;
	surf2Dread(&f1, source, idx_i, idx_j);
	surf2Dread(&f2, source, idx_i + 4, idx_j);
	surf2Dread(&f3, source, idx_i, idx_j + 1);
	surf2Dread(&f4, source, idx_i + 4, idx_j + 1);

	constexpr float three_std_dev = 3.0f * cSigmaR;

	auto acc = f1;
	int count = 1;
	if (fabsf(f1 - f2) <= three_std_dev)
	{
		acc += f2;
		++count;
	}
	if (fabsf(f1 - f3) <= three_std_dev)
	{
		acc += f3;
		++count;
	}
	if (fabsf(f1 - f4) <= three_std_dev)
	{
		acc += f4;
		++count;
	}

	surf2Dwrite(acc / count, destination, i * 4, j);
}

__global__ void createVertexMapKernel(cudaSurfaceObject_t depth_frame, cudaSurfaceObject_t vertex_map, glm::mat3 inv_cam_k, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float depth;
	surf2Dread(&depth, depth_frame, i * 4, j);

	glm::vec3 p(i + 0.5f, j + 0.5f, 1.0f);
	p.x *= scale;
	p.y *= scale;
	p = inv_cam_k * p;
	p *= depth;

	int idx = i * 16;
	surf2Dwrite(p.x, vertex_map, idx, j);
	surf2Dwrite(p.y, vertex_map, idx + 4, j);
	surf2Dwrite(p.z, vertex_map, idx + 8, j);
}

__global__ void createNormalMapKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t normal_map)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	glm::vec3 uv, u_1v, uv_1;
	int idx = i * 16;
	surf2Dread(&uv.x, vertex_map, idx, j);
	surf2Dread(&uv.y, vertex_map, idx + 4, j);
	surf2Dread(&uv.z, vertex_map, idx + 8, j);

	surf2Dread(&u_1v.x, vertex_map, idx + 16, j, cudaBoundaryModeClamp);
	surf2Dread(&u_1v.y, vertex_map, idx + 20, j, cudaBoundaryModeClamp);
	surf2Dread(&u_1v.z, vertex_map, idx + 24, j, cudaBoundaryModeClamp);

	surf2Dread(&uv_1.x, vertex_map, idx, j + 1, cudaBoundaryModeClamp);
	surf2Dread(&uv_1.y, vertex_map, idx + 4, j + 1, cudaBoundaryModeClamp);
	surf2Dread(&uv_1.z, vertex_map, idx + 8, j + 1, cudaBoundaryModeClamp);

	auto normal = glm::normalize(glm::cross(u_1v - uv, uv_1 - uv));
	surf2Dwrite(normal.x, normal_map, idx, j);
	surf2Dwrite(normal.y, normal_map, idx + 4, j);
	surf2Dwrite(normal.z, normal_map, idx + 8, j);
}

__global__ void oneHalfChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned short h_pixel;
	surf2Dread(&h_pixel, surface, i * 2, j);

	auto pixel = static_cast<unsigned char>(__half2float(h_pixel) * scale);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel);

	surf2Dwrite(pixel_w, window_content, i * 4, j);
}

__global__ void oneFloatChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float f_pixel;
	surf2Dread(&f_pixel, surface, i * 4, j);

	auto pixel = static_cast<unsigned char>(f_pixel * scale);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel) << 8;
	pixel_w = (pixel_w | pixel);

	surf2Dwrite(pixel_w, window_content, i * 4, j);
}

__global__ void fourFloatChannelToWindowContentKernel(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float r, g, b;
	int idx = i * 16;
	surf2Dread(&r, surface, idx, j);
	surf2Dread(&g, surface, idx + 4, j);
	surf2Dread(&b, surface, idx + 8, j);

	unsigned int pixel_w = (255) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(b * scale)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(g * scale)) << 8;
	pixel_w = (pixel_w | static_cast<unsigned char>(r * scale));

	surf2Dwrite(pixel_w, window_content, i * 4, j);
}

namespace kernel
{
	float applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		applyBilateralFilterKernel << <blocks, threads >> > (input, output);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(output_width / threads.x, output_height / threads.y);
		start.record();
		downSampleKernel << <blocks, threads >> > (input, output);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float createVertexMap(cudaSurfaceObject_t input_depth, cudaSurfaceObject_t output_vertex, const glm::mat3& inv_cam_k, int width, int height)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(width / threads.x, height / threads.y);
		start.record();
		createVertexMapKernel << <blocks, threads >> > (input_depth, output_vertex, inv_cam_k, 640 / width);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float computeNormalMap(CudaGridMap &vertex_map, CudaGridMap &normal_map)
	{
        auto dims = vertex_map.getGridDims();

		CudaEvent start, end;
		dim3 threads(8, 8);
        dim3 blocks(dims[0] / threads.x, dims[1] / threads.y);
		start.record();
		createNormalMapKernel << <blocks, threads >> > (vertex_map.getCudaSurfaceObject(), normal_map.getCudaSurfaceObject());
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float oneHalfChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		oneHalfChannelToWindowContentKernel << <blocks, threads >> > (surface, window_content, scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float oneFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		oneFloatChannelToWindowContentKernel << <blocks, threads >> > (surface, window_content, scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float fourFloatChannelToWindowContent(cudaSurfaceObject_t surface, cudaSurfaceObject_t window_content, float scale)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		fourFloatChannelToWindowContentKernel <<<blocks, threads >>> (surface, window_content, scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}