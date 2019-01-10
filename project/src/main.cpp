#include <glm/mat3x3.hpp>

#include "depth_map.h"
#include "cuda_grid_map.h"
#include "window.h"
#include "timer.h"
#include "measurement.cuh"
#include "data_helper.h"

int main()
{
    const bool use_kinect = false;
    const unsigned int frame_width = 640;
    const unsigned int frame_height = 480;
    
    // Hardcoded camera intrinsics.
    // Todo: Use an external configuration file to store and load the intrinsics (and any other configureations).
    const glm::mat3 camera_intrinsics(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
    const glm::mat3 inv_camera_intrinsics = glm::inverse(camera_intrinsics);

	//DepthMap depth_frame;
    cudaChannelFormatDesc raw_depth_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc vertex_and_normal_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    DepthMap raw_depth_map(frame_width, frame_height, raw_depth_desc);
    auto depth_map_pyramid = CudaGridMap::create3LayerPyramid(frame_width, frame_height, depth_desc);
    auto vertex_map_pyramid = CudaGridMap::create3LayerPyramid(frame_width, frame_height, vertex_and_normal_desc);
    auto normal_map_pyramid = CudaGridMap::create3LayerPyramid(frame_width, frame_height, vertex_and_normal_desc);

    if (!use_kinect)
    {
        // Load some example depth map.
        const std::string depth_frame_path = "frame.png";
        raw_depth_map.update(depth_frame_path);
    }
    //auto depth_pyramid = depth_frame.getPyramid();

	Window window = Window(use_kinect);
	Timer timer;
    float total_execution_time = 0.0;
	while (total_execution_time < 1e4)
	{
        if (use_kinect)
        {
            //Get depth frame from kinect.
            window.getKinectData(raw_depth_map);
        }

		auto total_kernel_time = 0.0f;
		timer.start();

		total_kernel_time += kernel::applyBilateralFilter(raw_depth_map.getCudaSurfaceObject(), depth_map_pyramid[0]->getCudaSurfaceObject());
		total_kernel_time += kernel::downSample(depth_map_pyramid[0]->getCudaSurfaceObject(), depth_map_pyramid[1]->getCudaSurfaceObject(), 320, 240);
		total_kernel_time += kernel::downSample(depth_map_pyramid[1]->getCudaSurfaceObject(), depth_map_pyramid[2]->getCudaSurfaceObject(), 160, 120);

		total_kernel_time += kernel::createVertexMap(depth_map_pyramid[0]->getCudaSurfaceObject(), vertex_map_pyramid[0]->getCudaSurfaceObject(), inv_camera_intrinsics, 640, 480);
        total_kernel_time += kernel::createVertexMap(depth_map_pyramid[1]->getCudaSurfaceObject(), vertex_map_pyramid[1]->getCudaSurfaceObject(), inv_camera_intrinsics, 320, 240);
        total_kernel_time += kernel::createVertexMap(depth_map_pyramid[2]->getCudaSurfaceObject(), vertex_map_pyramid[2]->getCudaSurfaceObject(), inv_camera_intrinsics, 160, 120);

		total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[0]->getCudaSurfaceObject(), normal_map_pyramid[0]->getCudaSurfaceObject(), 640, 480);
        total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[1]->getCudaSurfaceObject(), normal_map_pyramid[1]->getCudaSurfaceObject(), 320, 240);
        total_kernel_time += kernel::createNormalMap(vertex_map_pyramid[2]->getCudaSurfaceObject(), normal_map_pyramid[2]->getCudaSurfaceObject(), 160, 120);
		
        total_kernel_time += kernel::oneFloatChannelToWindowContent(depth_map_pyramid[0]->getCudaSurfaceObject() , window.get_content(), 0.01f);
		window.draw();

		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(total_kernel_time));

        total_execution_time += total_kernel_time;
	}

    for (int i = 0; i < 3; i++)
    {
        delete vertex_map_pyramid[i];
        delete normal_map_pyramid[i];
        delete depth_map_pyramid[i];
    }

	return 0;
}