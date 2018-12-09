#include "depth_frame.h"
#include "measurement.cuh"

int main()
{
	DepthFrame df;
	df.update("frame.png");

	auto depth_pyramid = df.getPyramid();

	kernel::applyBilateralFilter(df.getRaw(), depth_pyramid[0]);
	kernel::downSample(depth_pyramid[0], depth_pyramid[1], 320, 240);
	kernel::downSample(depth_pyramid[1], depth_pyramid[2], 160, 120);

	df.writePyramid();

	return 0;
}