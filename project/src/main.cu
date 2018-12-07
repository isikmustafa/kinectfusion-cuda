#include "depth_frame.h"
#include "cuda_event.h"
#include "measurement.cuh"

#include <iostream>

int main()
{
	DepthFrame df;
	df.update("frame.png");

	CudaEvent start, end;

	dim3 threads(8, 8);
	std::cout << "Kernel execution is started" << std::endl;
	auto depth_pyramid = df.getPyramid();

	start.record();
	applyBilateralFilter <<<dim3(640 / threads.x, 480 / threads.y), threads>>> (df.getRaw(), depth_pyramid[0]);
	downSample <<<dim3(640 / threads.x, 480 / threads.y), threads>>> (depth_pyramid[0], depth_pyramid[1]);
	downSample <<<dim3(320 / threads.x, 240 / threads.y), threads>>> (depth_pyramid[1], depth_pyramid[2]);
	end.record();
	end.synchronize();
	std::cout << "Kernel execution time: " << CudaEvent::calculateElapsedTime(start, end) << " ms" << std::endl;

	df.writePyramid();

	return 0;
}