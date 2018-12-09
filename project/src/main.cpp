#include "depth_frame.h"
#include "vertex_map.h"
#include "normal_map.h"
#include "measurement.cuh"

#include <glm/mat3x3.hpp>

int main()
{
	//Hardcoded camera intrinsics.
	glm::mat3 cam_k(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
	glm::mat3 inv_cam_k = glm::inverse(cam_k);

	DepthFrame df;
	VertexMap vm;
	NormalMap nm;
	df.update("frame.png");

	auto depth_pyramid = df.getPyramid();
	auto vertex_map_pyramid = vm.getPyramid();
	auto normal_map_pyramid = nm.getPyramid();

	kernel::applyBilateralFilter(df.getRaw(), depth_pyramid[0]);
	kernel::downSample(depth_pyramid[0], depth_pyramid[1], 320, 240);
	kernel::downSample(depth_pyramid[1], depth_pyramid[2], 160, 120);

	kernel::createVertexMap(depth_pyramid[0], vertex_map_pyramid[0], inv_cam_k, 640, 480);
	kernel::createVertexMap(depth_pyramid[1], vertex_map_pyramid[1], inv_cam_k, 320, 240);
	kernel::createVertexMap(depth_pyramid[2], vertex_map_pyramid[2], inv_cam_k, 160, 120);

	kernel::createNormalMap(vertex_map_pyramid[0], normal_map_pyramid[0], 640, 480);
	kernel::createNormalMap(vertex_map_pyramid[1], normal_map_pyramid[1], 320, 240);
	kernel::createNormalMap(vertex_map_pyramid[2], normal_map_pyramid[2], 160, 120);

	vm.writePyramid();
	df.writePyramid();
	nm.writePyramid();

	return 0;
}