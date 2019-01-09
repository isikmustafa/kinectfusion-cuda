#include <glm/mat3x3.hpp>

#include "depth_frame.h"
#include "vector_map.h"
#include "window.h"
#include "timer.h"
#include "measurement.cuh"


int main()
{
    const bool use_kinect = false;

	// Hardcoded camera intrinsics.
	// Todo: Use an external configuration file to store and load the intrinsics (and any other configureations).
	glm::mat3 camera_intrinsics(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
	glm::mat3 inv_camera_intrinsics = glm::inverse(camera_intrinsics);

	DepthFrame depth_frame;
	VectorMap vertex_map;
	VectorMap normal_map;
	
	Window window = Window(use_kinect);

    if (!use_kinect) 
    {
        // Load some example depth map.
        const std::string depth_frame_path = "frame.png";
        depth_frame.update(depth_frame_path);
    }

	auto depth_pyramid = depth_frame.getPyramid();
	auto vertex_map_pyramid = vertex_map.getPyramid();
	auto normal_map_pyramid = normal_map.getPyramid();

	Timer timer;
    float total_execution_time = 0.0;
	while (total_execution_time < 1e4)
	{
        if (use_kinect)
        {
            //Get depth frame from kinect.
            window.getKinectData(depth_frame);
        }

		auto total_kernel_time = 0.0f;
		timer.start();

		total_kernel_time += kernel::applyBilateralFilter(depth_frame.getRaw(), depth_pyramid[0]);
		total_kernel_time += kernel::downSample(depth_pyramid[0], depth_pyramid[1], 320, 240);
		total_kernel_time += kernel::downSample(depth_pyramid[1], depth_pyramid[2], 160, 120);

		total_kernel_time += kernel::createVertexMap(depth_pyramid[0], vertex_map_pyramid[0], inv_camera_intrinsics, 640, 480);
		total_kernel_time += kernel::createVertexMap(depth_pyramid[1], vertex_map_pyramid[1], inv_camera_intrinsics, 320, 240);
		total_kernel_time += kernel::createVertexMap(depth_pyramid[2], vertex_map_pyramid[2], inv_camera_intrinsics, 160, 120);

		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[0], normal_map_pyramid[0], 640, 480);
		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[1], normal_map_pyramid[1], 320, 240);
		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[2], normal_map_pyramid[2], 160, 120);
		
		total_kernel_time += kernel::oneFloatChannelToWindowContent(depth_pyramid[0], window.get_content(), 0.01f);
		window.draw();

		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(total_kernel_time));

        total_execution_time += total_kernel_time;
	}

	return 0;
}