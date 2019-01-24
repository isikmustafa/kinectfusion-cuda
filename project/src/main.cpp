#include "data_helper.h"
#include "sensor.h"
#include "depth_map.h"
#include "cuda_grid_map.h"
#include "window.h"
#include "timer.h"
#include "measurement.cuh"
#include "tsdf.cuh"
#include "raycast.cuh"
#include "grid_map_pyramid.h"
#include "general_helper.h"
#include "voxel_grid.h"
#include "rgbd_dataset.h"
#include "display.cuh"

int main()
{
	Sensor depth_sensor;
	VoxelGrid voxel_grid(3.0f, 512);
	RgbdDataset rgbd_dataset;
	rgbd_dataset.load("rgbd_dataset_freiburg1_xyz");

	constexpr bool use_kinect = false;
	constexpr unsigned int frame_width = 640;
	constexpr unsigned int frame_height = 480;
	constexpr unsigned int n_pyramid_layers = 3;

	cudaChannelFormatDesc raw_depth_desc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc vertex_and_normal_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	glm::vec3 light_dir(1.0f, 2.0f, 3.0f);
	DepthMap raw_depth_map(frame_width, frame_height, raw_depth_desc);
	CudaGridMap raw_depth_map_meters(frame_width, frame_height, depth_desc);
	CudaGridMap predicted_vertex_map(frame_width, frame_height, vertex_and_normal_desc);
	CudaGridMap predicted_normal_map(frame_width, frame_height, vertex_and_normal_desc);
	GridMapPyramid<CudaGridMap> depth_map_pyramid(frame_width, frame_height, n_pyramid_layers, depth_desc);
	GridMapPyramid<CudaGridMap> vertex_map_pyramid(frame_width, frame_height, n_pyramid_layers, vertex_and_normal_desc);
	GridMapPyramid<CudaGridMap> normal_map_pyramid(frame_width, frame_height, n_pyramid_layers, vertex_and_normal_desc);

	//Initialize grid.
	kernel::initializeGrid(voxel_grid.getStruct(), Voxel());

	Window window(use_kinect);
	Timer timer;
	while (true)
	{
		if (use_kinect)
		{
			//Get depth frame from kinect.
			window.getKinectData(raw_depth_map);
		}
		else if (!rgbd_dataset.isFinished())
		{
			auto next = rgbd_dataset.nextDepthAndPose();
			raw_depth_map.update(next.first);
			depth_sensor.setPose(next.second);
		}
        else
        {
            break;
        }

		auto total_kernel_time = 0.0f;
		timer.start();

		total_kernel_time += kernel::convertToDepthMeters(raw_depth_map, raw_depth_map_meters, 1.0f / 5000.0f);
		total_kernel_time += kernel::applyBilateralFilter(raw_depth_map_meters, depth_map_pyramid[0]);
		total_kernel_time += kernel::downSample(depth_map_pyramid[0], depth_map_pyramid[1]);
		total_kernel_time += kernel::downSample(depth_map_pyramid[1], depth_map_pyramid[2]);

		total_kernel_time += kernel::createVertexMap(depth_map_pyramid[0], vertex_map_pyramid[0], depth_sensor.getInverseIntr(0));
		total_kernel_time += kernel::createVertexMap(depth_map_pyramid[1], vertex_map_pyramid[1], depth_sensor.getInverseIntr(1));
		total_kernel_time += kernel::createVertexMap(depth_map_pyramid[2], vertex_map_pyramid[2], depth_sensor.getInverseIntr(2));

		total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[0], normal_map_pyramid[0]);
		total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[1], normal_map_pyramid[1]);
		total_kernel_time += kernel::computeNormalMap(vertex_map_pyramid[2], normal_map_pyramid[2]);

		total_kernel_time += kernel::fuse(raw_depth_map_meters, voxel_grid.getStruct(), depth_sensor);
		total_kernel_time += kernel::raycast(voxel_grid.getStruct(), depth_sensor, predicted_vertex_map, predicted_normal_map);

		total_kernel_time += kernel::fourFloatChannelToWindowContent(predicted_normal_map.getCudaSurfaceObject(), window, 255.0f);
		/*auto light_dir_eye = glm::normalize(glm::mat3(depth_sensor.getInversePose()) * light_dir);
		total_kernel_time += kernel::shadingToWindowContent(predicted_normal_map.getCudaSurfaceObject(), window, depth_sensor);*/
		window.draw();

		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(total_kernel_time));
	}

	writeSurface4x32("predicted.png", predicted_normal_map.getCudaArray(), 640, 480);
	writeSurface4x32("filtered.png", normal_map_pyramid[0].getCudaArray(), 640, 480);

	return 0;
}