#include "measurement.cuh"
#include "device_helper.cuh"

#include <iostream>

#include <cuda_fp16.h>
#include <glm/vec3.hpp>
#include "cuda_event.h"

//TODO: How do we decide?
constexpr float cSigmaS = 4.0f;
constexpr float cSigmaR = 0.25f;

__global__ void convertToDepthMetersKernel(cudaSurfaceObject_t raw_depth_map, cudaSurfaceObject_t raw_depth_map_meters, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	unsigned short h_depth;
	surf2Dread(&h_depth, raw_depth_map, i * 2, j);
	auto depth = __half2float(h_depth);

	//Convert depth value to value in meters
	surf2Dwrite(depth * scale, raw_depth_map_meters, i * 4, j);
}

__global__ void applyBilateralFilterKernel(cudaSurfaceObject_t raw_depth_map_meters, cudaSurfaceObject_t filtered_depth_map)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	constexpr int w_size = 7;

	//Do not change.
	constexpr int half_w_size = w_size / 2;
	constexpr float one_over_sigmasqr_s = 1.0f / (cSigmaS * cSigmaS);
	constexpr float one_over_sigmasqr_r = 1.0f / (cSigmaR * cSigmaR);

	float center, current;
	surf2Dread(&center, raw_depth_map_meters, i * 4, j);
	auto normalization = 0.0f;
	auto acc = 0.0f;
	for (int x = -half_w_size; x <= half_w_size; ++x)
	{
		for (int y = -half_w_size; y <= half_w_size; ++y)
		{
			surf2Dread(&current, raw_depth_map_meters, (i + x) * 4, j + y, cudaBoundaryModeClamp);

			auto s_dist_sqr = static_cast<float>(x * x + y * y);
			auto i_dist_sqr = (center - current);
			i_dist_sqr *= i_dist_sqr;
			auto factor = expf(-s_dist_sqr * one_over_sigmasqr_s - i_dist_sqr * one_over_sigmasqr_r);
			normalization += factor;

			acc += factor * current;
		}
	}

	surf2Dwrite(acc / normalization, filtered_depth_map, i * 4, j);
}

__global__ void downSampleKernel(cudaSurfaceObject_t depth_map, cudaSurfaceObject_t depth_map_downsampled)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	int idx_i = i * 8;
	int idx_j = j * 2;
	float f1, f2, f3, f4;
	surf2Dread(&f1, depth_map, idx_i, idx_j);
	surf2Dread(&f2, depth_map, idx_i + 4, idx_j);
	surf2Dread(&f3, depth_map, idx_i, idx_j + 1);
	surf2Dread(&f4, depth_map, idx_i + 4, idx_j + 1);

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

	surf2Dwrite(acc / count, depth_map_downsampled, i * 4, j);
}

__global__ void createVertexMapKernel(cudaSurfaceObject_t depth_map, cudaSurfaceObject_t vertex_map, glm::mat3 inv_cam_k, float scale)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	float depth;
	surf2Dread(&depth, depth_map, i * 4, j);

	if (device_helper::isDepthValid(depth))
	{
		glm::vec3 p(i + 0.5f, j + 0.5f, 1.0f);
		p.x *= scale;
		p.y *= scale;
		p = inv_cam_k * p;
		p *= depth;

		device_helper::writeVec3(p, vertex_map, i, j);
		device_helper::validate(vertex_map, i, j);
	}
	else
	{
		device_helper::invalidate(vertex_map, i, j);
	}
}

__global__ void createNormalMapKernel(cudaSurfaceObject_t vertex_map, cudaSurfaceObject_t normal_map)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

    auto normal = device_helper::computeNormal(vertex_map, i, j);
	device_helper::writeVec3(normal, normal_map, i, j);
}

namespace kernel
{
	float convertToDepthMeters(const DepthMap& raw_depth_map, const CudaGridMap& raw_depth_map_meters, float scale)
	{
		auto dims_input = raw_depth_map.getGridDims();
		auto dims_output = raw_depth_map_meters.getGridDims();

		if (dims_input != dims_output)
		{
			throw std::runtime_error("convertToDepthMeters: input and output surface objects are not of same size!");
		}

		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(dims_input[0] / threads.x, dims_input[1] / threads.y);
		start.record();
		convertToDepthMetersKernel << <blocks, threads >> > (raw_depth_map.getCudaSurfaceObject(), raw_depth_map_meters.getCudaSurfaceObject(), scale);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float applyBilateralFilter(const CudaGridMap& raw_depth_map_meters, const CudaGridMap& filtered_depth_map)
	{
		auto dims_input = raw_depth_map_meters.getGridDims();
		auto dims_output = filtered_depth_map.getGridDims();

		if (dims_input != dims_output)
		{
			throw std::runtime_error("applyBilateralFilter: input and output surface objects are not of same size!");
		}

		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(dims_input[0] / threads.x, dims_input[1] / threads.y);
		start.record();
		applyBilateralFilterKernel << <blocks, threads >> > (raw_depth_map_meters.getCudaSurfaceObject(), filtered_depth_map.getCudaSurfaceObject());
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float downSample(const CudaGridMap& depth_map, const CudaGridMap& depth_map_downsampled)
	{
		auto dims_input = depth_map.getGridDims();
		auto dims_output = depth_map_downsampled.getGridDims();

		if (dims_input[0] / dims_output[0] != 2 || dims_input[1] / dims_output[1] != 2)
		{
			throw std::runtime_error("downSample: output has to be half size of the input surface object!");
		}

		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(dims_output[0] / threads.x, dims_output[1] / threads.y);
		start.record();
		downSampleKernel << <blocks, threads >> > (depth_map.getCudaSurfaceObject(), depth_map_downsampled.getCudaSurfaceObject());
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float createVertexMap(const CudaGridMap& depth_map, const CudaGridMap& vertex_map, const glm::mat3& inv_cam_k)
	{
		auto dims_input = depth_map.getGridDims();
		auto dims_output = vertex_map.getGridDims();

		if (dims_input != dims_output)
		{
			throw std::runtime_error("createVertexMap: input and output surface objects are not of same size!");
		}

		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(dims_input[0] / threads.x, dims_input[1] / threads.y);
		start.record();
		createVertexMapKernel << <blocks, threads >> > (depth_map.getCudaSurfaceObject(), vertex_map.getCudaSurfaceObject(), inv_cam_k, 640 / dims_input[0]);
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}

	float computeNormalMap(const CudaGridMap& vertex_map, const CudaGridMap& normal_map)
	{
		auto dims_input = vertex_map.getGridDims();
		auto dims_output = normal_map.getGridDims();

		if (dims_input != dims_output)
		{
			throw std::runtime_error("computeNormalMap: input and output surface objects are not of same size!");
		}

		CudaEvent start, end;
		dim3 threads(8, 8);
        dim3 blocks(dims_input[0] / threads.x, dims_input[1] / threads.y);
		start.record();
		createNormalMapKernel << <blocks, threads >> > (vertex_map.getCudaSurfaceObject(), normal_map.getCudaSurfaceObject());
		end.record();
		end.synchronize();

		return CudaEvent::calculateElapsedTime(start, end);
	}
}