#include "depth_frame.h"
#include "vertex_map.h"
#include "normal_map.h"
#include "measurement.cuh"
#include "window.h"
#include "timer.h"

#include <glm/mat3x3.hpp>

int main()
{
	//Hardcoded camera intrinsics.
	glm::mat3 cam_k(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
	glm::mat3 inv_cam_k = glm::inverse(cam_k);

	DepthFrame df;
	VertexMap vm;
	NormalMap nm;
	Window window;
	df.update("frame.png");

	auto depth_pyramid = df.getPyramid();
	auto vertex_map_pyramid = vm.getPyramid();
	auto normal_map_pyramid = nm.getPyramid();

	Timer timer;
	while (true)
	{
		auto total_kernel_time = 0.0f;
		timer.start();

		total_kernel_time += kernel::applyBilateralFilter(df.getRaw(), depth_pyramid[0]);
		total_kernel_time += kernel::downSample(depth_pyramid[0], depth_pyramid[1], 320, 240);
		total_kernel_time += kernel::downSample(depth_pyramid[1], depth_pyramid[2], 160, 120);

		total_kernel_time += kernel::createVertexMap(depth_pyramid[0], vertex_map_pyramid[0], inv_cam_k, 640, 480);
		total_kernel_time += kernel::createVertexMap(depth_pyramid[1], vertex_map_pyramid[1], inv_cam_k, 320, 240);
		total_kernel_time += kernel::createVertexMap(depth_pyramid[2], vertex_map_pyramid[2], inv_cam_k, 160, 120);

		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[0], normal_map_pyramid[0], 640, 480);
		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[1], normal_map_pyramid[1], 320, 240);
		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[2], normal_map_pyramid[2], 160, 120);
		
		total_kernel_time += kernel::fourFloatChannelToWindowContent(normal_map_pyramid[0], window.get_content(), 255.0f);
		window.draw();

		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(total_kernel_time));
	}

	return 0;
}