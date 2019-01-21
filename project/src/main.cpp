#include "data_helper.h"
#include "sensor.h"
#include "depth_map.h"
#include "cuda_grid_map.h"
#include "window.h"
#include "timer.h"
#include "measurement.cuh"
#include "tsdf.cuh"
#include "grid_map_pyramid.h"
#include "voxel_grid.h"
#include "display.cuh"

int main()
{
	Sensor depth_sensor;
	VoxelGrid voxel_grid(5.0f, 512);

    constexpr bool use_kinect = false;
	constexpr unsigned int frame_width = 640;
	constexpr unsigned int frame_height = 480;
    constexpr unsigned int n_pyramid_layers = 3;

    cudaChannelFormatDesc raw_depth_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    cudaChannelFormatDesc vertex_and_normal_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

    DepthMap raw_depth_map(frame_width, frame_height, raw_depth_desc);
	CudaGridMap raw_depth_map_meters(frame_width, frame_height, depth_desc);
    GridMapPyramid<CudaGridMap> depth_map_pyramid(frame_width, frame_height, n_pyramid_layers, depth_desc);
    GridMapPyramid<CudaGridMap> vertex_map_pyramid(frame_width, frame_height, n_pyramid_layers, vertex_and_normal_desc);
    GridMapPyramid<CudaGridMap> normal_map_pyramid(frame_width, frame_height, n_pyramid_layers, vertex_and_normal_desc);

    if (!use_kinect)
    {
        // Load some example depth map.
        const std::string depth_frame_path = "frame.png";
        raw_depth_map.update(depth_frame_path);
    }

	Window window = Window(use_kinect);
	Timer timer;
    float total_execution_time = 0.0;
	while (total_execution_time < 3e5)
	{
        if (use_kinect)
        {
            //Get depth frame from kinect.
            window.getKinectData(raw_depth_map);
        }

		auto total_kernel_time = 0.0f;
		timer.start();

		total_kernel_time += kernel::convertToDepthMeters(raw_depth_map, raw_depth_map_meters, 1.0f /*1.0f / 5000.0f*/);
		total_kernel_time += kernel::applyBilateralFilter(raw_depth_map_meters, depth_map_pyramid[0]);
		total_kernel_time += kernel::downSample(depth_map_pyramid[0], depth_map_pyramid[1]);
		total_kernel_time += kernel::downSample(depth_map_pyramid[1], depth_map_pyramid[2]);

		total_kernel_time += kernel::createVertexMap(depth_map_pyramid[0], vertex_map_pyramid[0], depth_sensor.getInverseIntr());
        total_kernel_time += kernel::createVertexMap(depth_map_pyramid[1], vertex_map_pyramid[1], depth_sensor.getInverseIntr());
        total_kernel_time += kernel::createVertexMap(depth_map_pyramid[2], vertex_map_pyramid[2], depth_sensor.getInverseIntr());

		total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[0], normal_map_pyramid[0]);
        total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[1], normal_map_pyramid[1]);
        total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[2], normal_map_pyramid[2]);

		total_kernel_time += kernel::fuse(raw_depth_map_meters, voxel_grid.getStruct(), depth_sensor);
		
        total_kernel_time += kernel::oneFloatChannelToWindowContent(depth_map_pyramid[0].getCudaSurfaceObject() , window, 0.01f);
		window.draw();

		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(total_kernel_time));

        total_execution_time += total_kernel_time;
	}

	return 0;
}