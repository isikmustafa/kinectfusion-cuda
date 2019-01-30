#include "kinect_fusion.h"

#include <iostream>
#include <glm/gtx/transform.hpp>
#include <fstream>
#include "tsdf.cuh"
#include "raycast.cuh"
#include "measurement.cuh"
#include "general_helper.h"
#include "display.cuh"
#include "fps_camera.h"

KinectFusion::KinectFusion(KinectFusionConfig &kf_config, IcpConfig &icp_config)
	: m_config(kf_config)
	, m_icp_config(icp_config)
	, m_moving_sensor(kf_config.field_of_view)
	, m_fixed_sensor(kf_config.field_of_view)
	, m_voxel_grid(kf_config.voxel_grid_size, kf_config.voxel_grid_resolution)
	, m_icp_registrator(icp_config)
	, m_window(kf_config.use_kinect)
	, m_raw_depth_map(kf_config.frame_width, kf_config.frame_height, m_depth_desc_half)
	, m_raw_depth_map_meters(kf_config.frame_width, kf_config.frame_height, m_depth_desc_single)
	, m_depth_pyramid(kf_config.frame_width, kf_config.frame_height, icp_config.iters_per_layer.size(),
		m_depth_desc_single)
	, m_vertex_pyramid(kf_config.frame_width, kf_config.frame_height, icp_config.iters_per_layer.size(),
		m_vector_4d_desc)
	, m_predicted_vertex_pyramid(kf_config.frame_width, kf_config.frame_height, icp_config.iters_per_layer.size(),
		m_vector_4d_desc)
	, m_predicted_normal_pyramid(kf_config.frame_width, kf_config.frame_height, icp_config.iters_per_layer.size(),
		m_vector_4d_desc)
{
	m_fixed_sensor.setPose(m_config.static_viewpoint);
    m_fixed_sensor.setPose(m_config.initial_pose);
	kernel::initializeGrid(m_voxel_grid.getStruct(), Voxel());

	if (!m_config.use_kinect && !m_config.load_mode)
	{
		m_rgbd_dataset.load(m_config.dataset_dir);
	}
}

void KinectFusion::startTracking(int n_frames)
{
	initializeTracking();

	/*std::ofstream output_file;
	std::string iters_per_layer = std::to_string(m_icp_config.iters_per_layer[0]) + "."
		+ std::to_string(m_icp_config.iters_per_layer[1]) + "." + std::to_string(m_icp_config.iters_per_layer[2]);
	std::string file_name = "freiburg_dataset_icp_" + iters_per_layer + "_" + std::to_string(double(m_icp_config.distance_thresh)) + "_" + std::to_string(double(m_icp_config.angle_thresh)) + ".csv";
	output_file.open(file_name);*/

	for (m_current_frame_number = 1; m_current_frame_number <= n_frames; m_current_frame_number++)
	{
		m_timer.start();

		m_window.handleInput();
		m_keyboard_state = m_window.getKeyboardState();

		readNextDephtMap();
		depthFrameToVertexPyramid();
		raycastTsdf();
		computePose();
		fuseCurrentDepthToTSDF();
		visualizeCurrentModel();

		if (m_config.verbose && m_current_frame_number % 50 == 0)
		{
			printTimings();
		}
		if (m_keyboard_state.enter)
		{
			break;
		}

		updateWindowTitle();
		/*saveStream(output_file);*/
		m_fusion_time = 0.0f;
		m_preprocessing_time = 0.0f;
		m_display_time = 0.0f;
	}
	/*output_file.close();*/
}

void KinectFusion::initializeTracking()
{
	m_total_angle_error = 0.0f;
	m_total_distance_error = 0.0f;

	m_moving_sensor.setPose(m_config.initial_pose);

	if (m_config.use_kinect)
	{
		warmupKinect();
	}

	readNextDephtMap();
	m_functions_to_times["other"] += m_preprocessing_time +=
		kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale, m_config.use_kinect);
	m_functions_to_times["fuse"] += m_fusion_time +=
		kernel::fuse(m_raw_depth_map_meters, m_voxel_grid.getStruct(), m_moving_sensor);
}

void KinectFusion::warmupKinect()
{
	for (int i = 0; i < 10; i++)
	{
		m_window.getKinectData(m_raw_depth_map);
	}
}

void KinectFusion::readNextDephtMap()
{
	if (m_config.use_kinect)
	{
		m_window.getKinectData(m_raw_depth_map);
	}
	else if (!m_rgbd_dataset.isFinished())
	{
		m_raw_depth_map.update(m_rgbd_dataset.getNextDepthImagePath());
	}
}

void KinectFusion::depthFrameToVertexPyramid()
{
	m_functions_to_times["other"] += m_preprocessing_time +=
		kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale, m_config.use_kinect);

	m_functions_to_times["other"] += m_preprocessing_time +=
		kernel::applyBilateralFilter(m_raw_depth_map_meters, m_depth_pyramid[0]);

	m_functions_to_times["other"] += m_preprocessing_time +=
		kernel::createVertexMap(m_depth_pyramid[0], m_vertex_pyramid[0], m_moving_sensor.getInverseIntr(0));

	for (int i = 1; i < m_icp_config.iters_per_layer.size(); i++)
	{
		m_functions_to_times["other"] += m_preprocessing_time +=
			kernel::downSample(m_depth_pyramid[i - 1], m_depth_pyramid[i]);
		m_functions_to_times["other"] += m_preprocessing_time +=
			kernel::createVertexMap(m_depth_pyramid[i], m_vertex_pyramid[i],
			m_moving_sensor.getInverseIntr(i));
	}
}

void KinectFusion::raycastTsdf()
{
	for (int i = 0; i < m_icp_config.iters_per_layer.size(); i++)
	{
		m_functions_to_times["raycast" + std::to_string(i)] += m_raycast_time = kernel::raycast(m_voxel_grid.getStruct(), m_moving_sensor,
			m_predicted_vertex_pyramid[i], m_predicted_normal_pyramid[i], i);
	}
}

void KinectFusion::computePose()
{
	RigidTransform3D pose_estimate;
	pose_estimate = m_icp_registrator.computePose(m_vertex_pyramid, m_predicted_vertex_pyramid,
		m_predicted_normal_pyramid, m_moving_sensor);

	m_moving_sensor.setPose(pose_estimate.getTransformation());

	auto icp_execution_times = m_icp_registrator.getExecutionTimes();
	m_functions_to_times["ICP - Residual Construction"] += m_icp_r_time = icp_execution_times[0];
	m_functions_to_times["ICP - Solve"] += m_icp_s_time = icp_execution_times[1];

	if (!m_config.use_kinect && m_config.compute_pose_error)
	{
		computePoseError();
	}
}

void KinectFusion::computePoseError()
{
	constexpr float pi = 3.14159265358979323846f;

	glm::mat4x4 reference_pose = m_config.initial_pose *  m_rgbd_dataset.getInitialPoseInverse()
		* m_rgbd_dataset.getCurrentPose();
	auto pose_error = poseError(reference_pose, m_moving_sensor.getPose());

	m_total_angle_error += m_angle_error = pose_error.first;
	m_total_distance_error += m_distance_error = pose_error.second;

	/*if (pose_error.first > 5.0 || pose_error.second > 0.05)
	{
		float angle_error_in_degree = glm::degrees(pose_error.first);
		std::cout << "Frame Number: " << m_current_frame_number << " Angle Error :" << angle_error_in_degree;
		std::cout << " Distance Error: " << pose_error.second << std::endl;
	}

	if (m_current_frame_number % 50 == 0)
	{
		std::cout << "Average angle error: " << 180 * (m_total_angle_error / m_current_frame_number) / pi;
		std::cout << " Average distance error : " << m_total_distance_error / m_current_frame_number << std::endl;
	}*/
}

void KinectFusion::fuseCurrentDepthToTSDF()
{
	m_functions_to_times["fuse"] += m_fusion_time+= kernel::fuse(m_raw_depth_map_meters, m_voxel_grid.getStruct(), m_moving_sensor);
}

void KinectFusion::visualizeCurrentModel()
{
	if (m_config.use_static_view)
	{
		m_functions_to_times["raycast(Visualization)"] += m_raycast_time =
			kernel::raycast(m_voxel_grid.getStruct(), m_fixed_sensor, m_predicted_vertex_pyramid[0],
			m_predicted_normal_pyramid[0], 0);
	}

	if (m_config.use_shading)
	{
		m_functions_to_times["other"] += m_display_time += kernel::shadingToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(), m_window,
			m_config.use_static_view ? m_fixed_sensor : m_moving_sensor);
	}
	else
	{
		m_functions_to_times["other"] += m_display_time += kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(),
			m_window, glm::mat3x3((m_config.use_static_view ? m_fixed_sensor : m_moving_sensor).getInversePose()));
	}
}

void KinectFusion::saveNormalMapToFile(std::string suffix)
{
	std::string frame_name = std::to_string(m_current_frame_number) + suffix + ".png";
	writeSurface4x32(frame_name, m_predicted_normal_pyramid[0].getCudaArray(), 640, 480);
}

void KinectFusion::updateWindowTitle()
{
	m_window.draw();
	m_window.setWindowTitle("Frame: " + std::to_string(m_current_frame_number)
		+ "Total frame time: " + std::to_string(m_timer.getTime() * 1000.0));
}

void KinectFusion::printTimings()
{
	/*std::cout << "Average timings" << std::endl;
	for (const auto& function_time : m_functions_to_times)
	{
		std::cout << function_time.first << ": " << function_time.second / m_current_frame_number << std::endl;
	}
	std::cout << std::endl;*/
}

void KinectFusion::changeView()
{
	float angle;
	glm::vec3 rotation_axis, translation_vector;
	float input_position;
	float pi = 3.14159265358979323846f;
	while (true)
	{
		std::cout << "Please enter rotation angle in degrees: ";
		std::cin >> angle;
		std::cout << "Please enter rotation axis in vector: ";
		std::cin >> rotation_axis.x;
		std::cin >> rotation_axis.y;
		std::cin >> rotation_axis.z;
		std::cout << "Please enter translation vector: ";
		std::cin >> translation_vector.x;
		std::cin >> translation_vector.y;
		std::cin >> translation_vector.z;
		Sensor dummy_sensor = m_moving_sensor;
		dummy_sensor.setPose(glm::translate(glm::rotate(dummy_sensor.getPose(), -glm::radians(angle),
			rotation_axis), translation_vector));
		kernel::raycast(m_voxel_grid.getStruct(), dummy_sensor,
			m_predicted_vertex_pyramid[0], m_predicted_normal_pyramid[0], 0);

		kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(),
			m_window, glm::mat3x3(dummy_sensor.getInversePose()));
		updateWindowTitle();
	}
}

void KinectFusion::walk()
{
	FpsCamera fps_camera(m_moving_sensor.getPose());

	while (true)
	{
		m_timer.start();

		kernel::raycast(m_voxel_grid.getStruct(), m_moving_sensor, m_predicted_vertex_pyramid[0], m_predicted_normal_pyramid[0], 0);

		if (m_config.use_shading)
		{
			kernel::shadingToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(), m_window, m_moving_sensor);
		}
		else
		{
			kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(), m_window,
				glm::mat3x3(m_moving_sensor.getInversePose()));
		}

		updateWindowTitle();

		//Handle inputs.
		m_window.handleInput();
		auto keyboard_state = m_window.getKeyboardState();
		auto delta_time = m_timer.getTime();

		auto sensitivity = 0.08f * delta_time;
		auto x_disp = 0.0f;
		auto z_disp = 0.0f;

		if (keyboard_state.w)
		{
			z_disp += delta_time;
		}
		if (keyboard_state.s)
		{
			z_disp -= delta_time;
		}
		if (keyboard_state.d)
		{
			x_disp += delta_time;
		}
		if (keyboard_state.a)
		{
			x_disp -= delta_time;
		}

		//Update camera's position and orientation.
		if (x_disp != 0.0f || z_disp != 0.0f)
		{
			fps_camera.move(x_disp, z_disp);
		}

		auto mouse_state = m_window.getMouseState();
		if (mouse_state.pressed)
		{
			auto offset = (mouse_state.previous_position - mouse_state.current_position);
			auto offset_x = offset.x * sensitivity;
			auto offset_y = offset.y * sensitivity;

			if (offset_x != 0.0f || offset_y != 0.0f)
			{
				fps_camera.rotate(-offset_x, offset_y);
			}
		}

		m_moving_sensor.setPose(fps_camera.getPose());
	}
}

void KinectFusion::saveTSDF(std::string file_name)
{
    m_voxel_grid.saveVoxelGrid(file_name);
}

void KinectFusion::loadTSDF(std::string file_name)
{
    m_voxel_grid.loadVoxelGrid(file_name);
}

void KinectFusion::saveStream(std::ofstream &output_file)
{
	std::string iters_per_layer = std::to_string(m_icp_config.iters_per_layer[0]) + "."
		+ std::to_string(m_icp_config.iters_per_layer[1]) + "." + std::to_string(m_icp_config.iters_per_layer[2]);
	output_file << iters_per_layer << ", " << m_icp_config.distance_thresh << ", " << m_icp_config.angle_thresh << ", " 
		<< m_distance_error << ", " << m_angle_error 
		<< m_raycast_time << ", " << m_icp_r_time << "," << m_icp_s_time <<", " <<m_fusion_time << ", " 
		<< m_preprocessing_time << ", "<<m_display_time  <<  "\n";

}