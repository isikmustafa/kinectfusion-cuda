#include "measurement.cuh"
#include "cuda_event.h"

#include <cuda_fp16.h>
#include <iostream>

__global__ void applyBilateralFilterKernel(cudaSurfaceObject_t raw, cudaSurfaceObject_t filtered)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;

	//Parameters of the bilateral filter.
	constexpr int w_size = 9;
	constexpr float sigma_s = 1.0f;
	constexpr float sigma_i = 1.0f;

	//Do not change. These are dependent to above.
	constexpr int half_w_size = w_size / 2;
	constexpr float one_over_sigmasqr_s = 1.0f / (sigma_s * sigma_s);
	constexpr float one_over_sigmasqr_i = 1.0f / (sigma_i * sigma_i);

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
			auto i_dist_sqr = (center - current) * (center - current);
			auto factor = expf(-s_dist_sqr * one_over_sigmasqr_s) * expf(-i_dist_sqr * one_over_sigmasqr_i);
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

	auto diff1 = f2 - f1;
	auto diff2 = f3 - f1;
	auto diff3 = f4 - f1;
	auto variance = (diff1 * diff1 + diff2 * diff2 + diff3 * diff3) * 0.33333f;
	auto three_std_dev = 3.0f * sqrtf(variance);

	auto acc = f1;
	int count = 1;
	if (fabsf(diff1) <= three_std_dev)
	{
		acc += f2;
		++count;
	}
	if (fabsf(diff2) <= three_std_dev)
	{
		acc += f3;
		++count;
	}
	if (fabsf(diff3) <= three_std_dev)
	{
		acc += f4;
		++count;
	}

	surf2Dwrite(acc / count, destination, i * 4, j);
}

namespace kernel
{
	void applyBilateralFilter(cudaSurfaceObject_t input, cudaSurfaceObject_t output)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(640 / threads.x, 480 / threads.y);
		start.record();
		applyBilateralFilterKernel <<<blocks, threads>>> (input, output);
		end.record();
		end.synchronize();
		std::cout << "applyBilateralFilter execution time: " << CudaEvent::calculateElapsedTime(start, end) << " ms" << std::endl;
	}

	void downSample(cudaSurfaceObject_t input, cudaSurfaceObject_t output, int output_width, int output_height)
	{
		CudaEvent start, end;
		dim3 threads(8, 8);
		dim3 blocks(output_width / threads.x, output_height / threads.y);
		start.record();
		downSampleKernel <<<blocks, threads>>> (input, output);
		end.record();
		end.synchronize();
		std::cout << "downSample execution time: " << CudaEvent::calculateElapsedTime(start, end) << " ms" << std::endl;
	}
}