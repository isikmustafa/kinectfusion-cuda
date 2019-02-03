#pragma once
#include <string>
#include <vector>
#include <unordered_map>
#include "glm_macro.h"
#include <glm/glm.hpp>

#include "sensor.h"
#include "voxel_grid.h"
#include "icp.h"
#include "depth_map.h"
#include "cuda_grid_map.h"
#include "rgbd_dataset.h"
#include "window.h"
#include "timer.h"

struct KinectFusionConfig
{
    bool use_kinect;
    bool verbose;
    bool use_static_view;
    bool compute_pose_error;
	bool use_shading;
    bool load_mode;
    std::string dataset_dir;
    unsigned int frame_width;
    unsigned int frame_height;
    float depth_scale;
    float field_of_view;
    float fusion_voxel_grid_size;
    float fusion_voxel_grid_n;
	float visualization_voxel_grid_size;
	float visualization_voxel_grid_n;
    glm::mat4 static_viewpoint;
    glm::mat4 initial_pose;
};

class KinectFusion
{
public:
    KinectFusion(KinectFusionConfig &kf_config, IcpConfig &icp_config);

    void startTracking(int n_frames);
	void changeView();
	void walk();
    void saveTSDF(std::string file_name);
    void loadTSDF(std::string file_name);

private:
    // Constants
    const cudaChannelFormatDesc m_depth_desc_half = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaChannelFormatDesc m_depth_desc_single = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    const cudaChannelFormatDesc m_vector_4d_desc = cudaCreateChannelDesc(32, 32, 32, 32, cudaChannelFormatKindFloat);
    
    // Configuration
    KinectFusionConfig m_config;
    IcpConfig m_icp_config;
   
    // Subsystems
    Sensor m_moving_sensor;
    Sensor m_fixed_sensor;
    VoxelGrid m_voxel_grid_fusion;
	VoxelGrid m_voxel_grid_visualization;
    ICP m_icp_registrator;
    RgbdDataset m_rgbd_dataset;
    Window m_window;
    Timer m_timer;
    
    // Data
    DepthMap m_raw_depth_map;
    CudaGridMap m_raw_depth_map_meters;
    GridMapPyramid<CudaGridMap> m_depth_pyramid;
    GridMapPyramid<CudaGridMap> m_vertex_pyramid;
    GridMapPyramid<CudaGridMap> m_predicted_vertex_pyramid;
    GridMapPyramid<CudaGridMap> m_predicted_normal_pyramid;

    // Other
    float m_total_angle_error;
    float m_total_distance_error;

    int m_current_frame_number;

	//Timings
	std::unordered_map<std::string, float> m_functions_to_times;

	float m_distance_error;
	float m_angle_error;

	//Inputs
	KeyboardState m_keyboard_state;

private:
    void initializeTracking();
    void warmupKinect();
    void readNextDephtMap();
	void depthFrameToVertexPyramid();
	void raycastTsdf();
	void computePose();
    void computePoseError();
	void fuseCurrentDepthToTSDF();
	void visualizeCurrentModel();
    void saveNormalMapToFile(std::string suffix);
    void updateWindowTitle();
	void printTimings();
	void saveStream(std::ofstream &output_file);
};