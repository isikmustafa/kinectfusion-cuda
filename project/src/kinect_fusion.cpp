#include "kinect_fusion.h"

#include <iostream>

#include "tsdf.cuh"
#include "raycast.cuh"
#include "measurement.cuh"
#include "general_helper.h"
#include "display.cuh"
#include <glm/gtx/transform.hpp>

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
    kernel::initializeGrid(m_voxel_grid.getStruct(), Voxel());

    if (!m_config.use_kinect)
    {
        m_rgbd_dataset.load(m_config.dataset_dir);
    }
}

KinectFusion::~KinectFusion()
{
}

void KinectFusion::startTracking(int n_frames)
{
    initializeTracking();

    for (m_current_frame_number = 1; m_current_frame_number <= n_frames; m_current_frame_number++)
    {
        m_timer.start();

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

        updateWindowTitle();
    }
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
	m_functions_to_times["other"] += 
		kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale);
	m_functions_to_times["fuse"] += 
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
	m_functions_to_times["other"] +=
		kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale);

	m_functions_to_times["other"] +=
		kernel::applyBilateralFilter(m_raw_depth_map_meters, m_depth_pyramid[0]);

	m_functions_to_times["other"] +=
		kernel::createVertexMap(m_depth_pyramid[0], m_vertex_pyramid[0], m_moving_sensor.getInverseIntr(0));

    for (int i = 1; i < m_icp_config.iters_per_layer.size(); i++)
    {
		m_functions_to_times["other"] += kernel::downSample(m_depth_pyramid[i - 1], m_depth_pyramid[i]);
		m_functions_to_times["other"] += kernel::createVertexMap(m_depth_pyramid[i], m_vertex_pyramid[i],
            m_moving_sensor.getInverseIntr(i));
    }
}

void KinectFusion::raycastTsdf()
{
    for (int i = 0; i < m_icp_config.iters_per_layer.size(); i++)
    {
		m_functions_to_times["raycast" + std::to_string(i)] += kernel::raycast(m_voxel_grid.getStruct(), m_moving_sensor,
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
	m_functions_to_times["ICP - Residual Construction"] += icp_execution_times[0];
	m_functions_to_times["ICP - Solve"] += icp_execution_times[1];

    if (!m_config.use_kinect && m_config.compute_pose_error)
    {
        computePoseError();
    }
}

void KinectFusion::computePoseError()
{
    static constexpr float pi = 3.14159265358979323846f;

    glm::mat4x4 reference_pose = m_config.initial_pose *  m_rgbd_dataset.getInitialPoseInverse()
        * m_rgbd_dataset.getCurrentPose();
    auto pose_error = poseError(reference_pose, m_moving_sensor.getPose());

    m_total_angle_error += pose_error.first;
    m_total_distance_error += pose_error.second;
    
    if (pose_error.first > 5.0 || pose_error.second > 0.05)
    {
        float angle_error_in_degree = 180.0f * pose_error.first / pi;
        std::cout << "Frame Number: " << m_current_frame_number << " Angle Error :" << angle_error_in_degree;
        std::cout << " Distance Error: " << pose_error.second << std::endl;
    }

    if (m_current_frame_number % 50 == 0)
    {
        std::cout << "Average angle error: " << 180 * (m_total_angle_error / m_current_frame_number) / pi;
        std::cout << " Average distance error : " << m_total_distance_error / m_current_frame_number << std::endl;
    }
}

void KinectFusion::fuseCurrentDepthToTSDF()
{
	m_functions_to_times["fuse"] += kernel::fuse(m_raw_depth_map_meters, m_voxel_grid.getStruct(), m_moving_sensor);
}

void KinectFusion::visualizeCurrentModel()
{
    if (m_config.use_static_view)
    {
		m_functions_to_times["raycast(Visualization)"] += kernel::raycast(m_voxel_grid.getStruct(), m_fixed_sensor, m_predicted_vertex_pyramid[0],
            m_predicted_normal_pyramid[0], 0);
    }

	m_functions_to_times["other"] += kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(),
        m_window, glm::mat3x3(m_moving_sensor.getInversePose()));
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
	std::cout << "Average timings" << std::endl;
	for (const auto& function_time : m_functions_to_times)
	{
		std::cout << function_time.first << ": " << function_time.second / m_current_frame_number << std::endl;
	}
	std::cout << std::endl;
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
		dummy_sensor.setPose(glm::translate(glm::rotate(dummy_sensor.getPose(), -pi / (180.0f / angle),
			rotation_axis), translation_vector));
		kernel::raycast(m_voxel_grid.getStruct(), dummy_sensor,
			m_predicted_vertex_pyramid[0], m_predicted_normal_pyramid[0], 0);

		kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(),
			m_window, glm::mat3x3(dummy_sensor.getInversePose()));
		updateWindowTitle();
	}
}