#include <iostream>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/transform.hpp>

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
#include "icp.h"

int main()
{
	// Configuration
	constexpr bool use_kinect = false;
	const std::string dataset_dir = "rgbd_dataset_freiburg1_xyz";
	constexpr auto depth_scale = 1.0f / (use_kinect ? 1000.0f : 5000.0f);
	constexpr unsigned int width = 640;
	constexpr unsigned int height = 480;

	std::vector<unsigned int> icp_iters_per_layer = { 10, 4, 2 }; // high -> low resolution
	const int n_pyramid_layers = icp_iters_per_layer.size();
	constexpr float icp_distance_thresh = 0.4f; // meters
	constexpr float pi = 3.14159265358979323846f;
	constexpr float icp_angle_thresh = 7.0f * pi / 18.0f;

    glm::mat4x4 static_viewpoint = glm::rotate(- pi / 36.0f, glm::vec3(1.0f, 0.0f, 0.0f));
    static_viewpoint[3] = glm::vec4(0.0f, -0.2f, -1.8f, 1.0f);

    glm::mat4x4 initial_pose(1.0f);
    initial_pose[3] = glm::vec4(0.0, 0.0, -1.0, 1.0);

	// Initialization
	Sensor moving_sensor(use_kinect ? 571.0f : 525.0f);
	Sensor fixed_sensor(use_kinect ? 571.0f : 525.0f);

	fixed_sensor.setPose(static_viewpoint);

	VoxelGrid voxel_grid(3.0f, 512);
	kernel::initializeGrid(voxel_grid.getStruct(), Voxel());

	ICP icp_registrator(icp_iters_per_layer, width, height, icp_distance_thresh, icp_angle_thresh);

	cudaChannelFormatDesc depth_desc_half = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc depth_desc_single = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
	cudaChannelFormatDesc vector_4d_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);

	DepthMap raw_depth_map(width, height, depth_desc_half);
	CudaGridMap raw_depth_map_meters(width, height, depth_desc_single);
	GridMapPyramid<CudaGridMap> depth_pyramid(width, height, n_pyramid_layers, depth_desc_single);
	GridMapPyramid<CudaGridMap> vertex_pyramid(width, height, n_pyramid_layers, vector_4d_desc);
	GridMapPyramid<CudaGridMap> predicted_vertex_pyramid(width, height, n_pyramid_layers, vector_4d_desc);
	GridMapPyramid<CudaGridMap> predicted_normal_pyramid(width, height, n_pyramid_layers, vector_4d_desc);

	RgbdDataset rgbd_dataset;
	if constexpr (!use_kinect)
	{
		rgbd_dataset.load(dataset_dir);
	}

	Window window(use_kinect);
	Timer timer;

	std::pair<std::string, glm::mat4> next;
	glm::mat4x4 first_pose_inverse;
	if constexpr (!use_kinect)
	{
		next = rgbd_dataset.nextDepthAndPose();
		raw_depth_map.update(next.first);
		first_pose_inverse = glm::inverse(next.second);
		raw_depth_map.update(next.first);
	}
	else
	{
		window.getKinectData(raw_depth_map);
	}

    moving_sensor.setPose(initial_pose);
	kernel::convertToDepthMeters(raw_depth_map, raw_depth_map_meters, depth_scale);
	kernel::fuse(raw_depth_map_meters, voxel_grid.getStruct(), moving_sensor);

	RigidTransform3D pose_estimate;
	pose_estimate.rot_mat = glm::mat3x3(moving_sensor.getPose());
	pose_estimate.transl_vec = moving_sensor.getPosition();

	// Register and fuse all subsequent frames
	int frame_number = 1;
	float total_angle_error = 0.0f;
	float total_distance_error = 0.0f;
	while (true)
	{
		// Get the new depth frame
		if (use_kinect)
		{
			//Get depth frame from kinect.
			window.getKinectData(raw_depth_map);
		}
		else if (!rgbd_dataset.isFinished())
		{
			next = rgbd_dataset.nextDepthAndPose();
			raw_depth_map.update(next.first);
		}
		else
		{
			break;
		}

		auto kernel_time = 0.0f;
		timer.start();

		// Process the new depth frame
		kernel_time += kernel::convertToDepthMeters(raw_depth_map, raw_depth_map_meters, depth_scale);
		kernel_time += kernel::applyBilateralFilter(raw_depth_map_meters, depth_pyramid[0]);
		kernel_time += kernel::downSample(depth_pyramid[0], depth_pyramid[1]);
		kernel_time += kernel::downSample(depth_pyramid[1], depth_pyramid[2]);

		kernel_time += kernel::createVertexMap(depth_pyramid[0], vertex_pyramid[0], moving_sensor.getInverseIntr(0));
		kernel_time += kernel::createVertexMap(depth_pyramid[1], vertex_pyramid[1], moving_sensor.getInverseIntr(1));
		kernel_time += kernel::createVertexMap(depth_pyramid[2], vertex_pyramid[2], moving_sensor.getInverseIntr(2));

		// Raycast vertex maps from current TSDF model
		auto previous_inverse_sensor_rotation = glm::mat3(moving_sensor.getInversePose());

		double raycast_times[3];

		kernel_time += raycast_times[0] = kernel::raycast(voxel_grid.getStruct(), moving_sensor, predicted_vertex_pyramid[0],
			predicted_normal_pyramid[0], 0);

		kernel_time += raycast_times[1] = kernel::raycast(voxel_grid.getStruct(), moving_sensor, predicted_vertex_pyramid[1],
			predicted_normal_pyramid[1], 1);

		kernel_time += raycast_times[2] = kernel::raycast(voxel_grid.getStruct(), moving_sensor, predicted_vertex_pyramid[2],
			predicted_normal_pyramid[2], 2);

		// Compute the camera pose for the new frame
		pose_estimate = icp_registrator.computePose(vertex_pyramid, predicted_vertex_pyramid, predicted_normal_pyramid,
			pose_estimate, moving_sensor);

		moving_sensor.setPose(pose_estimate.getTransformation());

		auto icp_execution_times = icp_registrator.getExecutionTimes();
		kernel_time += icp_execution_times[0] + icp_execution_times[1];

		if constexpr (!use_kinect)
		{
			auto pose_error = poseError(initial_pose *  first_pose_inverse * next.second,
				pose_estimate.getTransformation());

			total_angle_error += pose_error.first;
			total_distance_error += pose_error.second;
			if (pose_error.first > 5.0 || pose_error.second > 0.05)
			{
				std::cout << "Frame Number: " << frame_number << " Angle Error :" << 180 * pose_error.first / pi
					<< " Distance Error: " << pose_error.second << std::endl;
			}

			//At every 50th frame, print average errors.
			if (frame_number % 50 == 0)
			{
				std::cout << "Average angle error: " << 180 * (total_angle_error / frame_number) / pi;
				std::cout << " Average distance error : " << total_distance_error / frame_number << std::endl;
			}
		}

		// Fuse the new frame into the TSDF model
		kernel_time += kernel::fuse(raw_depth_map_meters, voxel_grid.getStruct(), moving_sensor);

		// Raycast vertex and normal maps from a fixed view
		kernel_time += kernel::raycast(voxel_grid.getStruct(), fixed_sensor, predicted_vertex_pyramid[0],
			predicted_normal_pyramid[0], 0);

		kernel_time += kernel::normalMapToWindowContent(predicted_normal_pyramid[0].getCudaSurfaceObject(),
			window, previous_inverse_sensor_rotation);

		//kernel_time += kernel::oneFloatChannelToWindowContent(depth_pyramid[0].getCudaSurfaceObject(), window, 128.0f);

		//glm::vec3 light_dir(1.0f, 2.0f, 3.0f);
		//auto light_dir_eye = glm::normalize(glm::mat3(moving_sensor.getInversePose()) * light_dir);
		//kernel_time += kernel::shadingToWindowContent(predicted_normal_map.getCudaSurfaceObject(), window, moving_sensor);

		window.draw();
		window.setWindowTitle("Total frame time: " + std::to_string(timer.getTime() * 1000.0) +
			" , Total kernel execution time: " + std::to_string(kernel_time));

		/*std::cout << "Raycast(1): " << raycast_times[0] << std::endl;
		std::cout << "Raycast(2): " << raycast_times[1] << std::endl;
		std::cout << "Raycast(3): " << raycast_times[2] << std::endl;

		std::cout << "ICP execution time(1): " << icp_execution_times[0] << std::endl;
		std::cout << "ICP execution time(2): " << icp_execution_times[1] << std::endl << std::endl;*/
		
        /*if(frame_number>0 && frame_number<20)
		{
			std::string frame_name = std::to_string(frame_number)+"_predicted_.png";
			std::cout << frame_name<<std::endl;
			writeSurface4x32(frame_name, predicted_normal_pyramid[0].getCudaArray(), 640, 480);
		}*/
		frame_number++;
	}


    std::getchar();
	return 0;
}