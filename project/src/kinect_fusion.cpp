#include "kinect_fusion.h"

#include <iostream>

#include "tsdf.cuh"
#include "raycast.cuh"
#include "measurement.cuh"
#include "general_helper.h"
#include "display.cuh"

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
        float kernel_time = 0.0f;
        m_timer.start();

        readNextDephtMap();
        kernel_time += depthFrameToVertexPyramid();
        kernel_time += raycastVertexAndNormalPyramid();
        kernel_time += computePose();
        kernel_time = fuseCurrentDepthToTSDF();
        
        kernel_time = visualizeCurrentModel();
        updateWindowTitle(kernel_time);
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
    kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale);
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

float KinectFusion::depthFrameToVertexPyramid()
{
    float kernel_time = 0.0f;
    kernel_time += kernel::convertToDepthMeters(m_raw_depth_map, m_raw_depth_map_meters, m_config.depth_scale);
    kernel_time += kernel::applyBilateralFilter(m_raw_depth_map_meters, m_depth_pyramid[0]);
    kernel_time += kernel::createVertexMap(m_depth_pyramid[0], m_vertex_pyramid[0], m_moving_sensor.getInverseIntr(0));

    for (int i = 1; i < m_icp_config.iters_per_layer.size(); i++)
    {
        kernel_time += kernel::downSample(m_depth_pyramid[i - 1], m_depth_pyramid[i]);
        kernel_time += kernel::createVertexMap(m_depth_pyramid[i], m_vertex_pyramid[i], 
            m_moving_sensor.getInverseIntr(i));
    }

    return kernel_time;
}

float KinectFusion::raycastVertexAndNormalPyramid()
{
    float kernel_time = 0.0f;

    for (int i = 0; i < m_icp_config.iters_per_layer.size(); i++)
    {
        float raycast_time;
        kernel_time += raycast_time = kernel::raycast(m_voxel_grid.getStruct(), m_moving_sensor, 
            m_predicted_vertex_pyramid[i], m_predicted_normal_pyramid[i], i);
        
        if (m_config.verbose)
        {
            std::cout << "Raycast(" + std::to_string(i) + "): " << raycast_time << std::endl;
        }
    }

    return kernel_time;
}

float KinectFusion::computePose()
{
    float kernel_time;
    RigidTransform3D pose_estimate;
    pose_estimate = m_icp_registrator.computePose(m_vertex_pyramid, m_predicted_vertex_pyramid,
        m_predicted_normal_pyramid, m_moving_sensor);

    m_moving_sensor.setPose(pose_estimate.getTransformation());

    auto icp_execution_times = m_icp_registrator.getExecutionTimes();
    kernel_time = icp_execution_times[0] + icp_execution_times[1];

    if (m_config.verbose)
    {
        std::cout << "ICP execution time(1): " << icp_execution_times[0] << std::endl;
        std::cout << "ICP execution time(2): " << icp_execution_times[1] << std::endl << std::endl;
    }

    if (!m_config.use_kinect && m_config.compute_pose_error)
    {
        computePoseError();
    }

    return kernel_time;
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

float KinectFusion::fuseCurrentDepthToTSDF()
{
    float kernel_time;
    kernel_time = kernel::fuse(m_raw_depth_map_meters, m_voxel_grid.getStruct(), m_moving_sensor);
    return kernel_time;
}

float KinectFusion::visualizeCurrentModel()
{
    float kernel_time = 0.0f;

    if (m_config.use_static_view)
    {
        kernel_time += kernel::raycast(m_voxel_grid.getStruct(), m_fixed_sensor, m_predicted_vertex_pyramid[0],
            m_predicted_normal_pyramid[0], 0);
    }

    kernel_time += kernel::normalMapToWindowContent(m_predicted_normal_pyramid[0].getCudaSurfaceObject(),
        m_window, glm::mat3x3(m_moving_sensor.getInversePose()));

    return kernel_time;
}

void KinectFusion::saveNormalMapToFile(std::string suffix)
{
    std::string frame_name = std::to_string(m_current_frame_number) + suffix + ".png";
    writeSurface4x32(frame_name, m_predicted_normal_pyramid[0].getCudaArray(), 640, 480);
}

void KinectFusion::updateWindowTitle(float kernel_time)
{
    m_window.draw();
    m_window.setWindowTitle("Frame: " + std::to_string(m_current_frame_number) 
        + "Total frame time: " + std::to_string(m_timer.getTime() * 1000.0) 
        + " , Total kernel execution time: " + std::to_string(kernel_time));
}
