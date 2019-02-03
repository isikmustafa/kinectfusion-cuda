#include "glm_macro.h"
#include <glm/gtx/transform.hpp>
#include <glm/glm.hpp>

#include "kinect_fusion.h"

int main()
{
    static constexpr float pi = 3.14159265358979323846f;

	KinectFusionConfig dataset_config;
    dataset_config.use_kinect = false;
    dataset_config.verbose = false;
    dataset_config.compute_pose_error = true;
    dataset_config.use_static_view = false;
	dataset_config.load_mode = false;
	dataset_config.use_shading = false;
    dataset_config.dataset_dir = "rgbd_dataset_freiburg1_rpy";
    dataset_config.depth_scale = 1.0f / 5000.0f;
    dataset_config.frame_width = 640;
    dataset_config.frame_height = 480;
    dataset_config.field_of_view = 525.0f;
	dataset_config.fusion_voxel_grid_size = 8.0f;
	dataset_config.fusion_voxel_grid_n = 256;
	dataset_config.visualization_voxel_grid_size = 3.0f;
	dataset_config.visualization_voxel_grid_n = 512;
    dataset_config.static_viewpoint = glm::rotate(- pi / 36.0f, glm::vec3(1.0f, 0.0f, 0.0f));
    dataset_config.static_viewpoint[3] = glm::vec4(0.0f, -0.2f, -1.8f, 1.0f);
    dataset_config.initial_pose = glm::mat4x4(1.0f);
    //dataset_config.initial_pose[3] = glm::vec4(1.0, 0.0, -3.0, 1.0);

	KinectFusionConfig kinect_config;
    kinect_config.use_kinect = true;
    kinect_config.verbose = false;
	kinect_config.compute_pose_error = false;
    kinect_config.use_static_view = false;
    kinect_config.load_mode = false;
	kinect_config.use_shading = false;
    kinect_config.depth_scale = 1.0f / 1000.0f;
    kinect_config.frame_width = 640;
    kinect_config.frame_height = 480;
    kinect_config.field_of_view = 571.0f;
	kinect_config.fusion_voxel_grid_size = 12.0f;
	kinect_config.fusion_voxel_grid_n = 256;
	kinect_config.visualization_voxel_grid_size = 4.0f;
	kinect_config.visualization_voxel_grid_n = 512;
    kinect_config.static_viewpoint = glm::rotate(-pi / 36.0f, glm::vec3(1.0f, 0.0f, 0.0f));
    kinect_config.static_viewpoint[3] = glm::vec4(0.0f, -0.2f, 0.0f, 1.0f);
    kinect_config.initial_pose = glm::mat4x4(1.0f);
    //kinect_config.initial_pose[3] = glm::vec4(0.0, 0.0, -0.5, 1.0);

	IcpConfig icp_config;
    icp_config.iters_per_layer = {10, 5, 3};
    icp_config.width = 640;
    icp_config.height = 480;
    icp_config.distance_thresh = 0.4f;
    icp_config.angle_thresh = 4.0f * pi / 18.0f;
	icp_config.iteration_stop_thresh_angle = glm::radians(0.1f);
	icp_config.iteration_stop_thresh_distance = 0.01f;

    KinectFusion awesome(dataset_config, icp_config);
    awesome.startTracking(790);
    //awesome.loadTSDF("kitchen_detail_1.bin");
    //awesome.saveTSDF("kitchen.bin");
    awesome.walk();

	/*std::vector<unsigned int> iters_per_layer[4] = { { 1, 1, 1 }, { 10, 4, 2 }, { 10, 0, 0 }, { 0, 10, 0 } };
	float distance_thresh[3] = { 0.2f, 0.4f, 0.6f };
	float angle_thresh[3] = { 3.0f * (pi / 18.0f), 6.0f * (pi / 18.0f), 9.0f * (pi / 18.0f) };

	for (int layer_index = 0; layer_index < 4; layer_index++)
	{
		for (int dist_index = 0; dist_index < 3; dist_index++)
		{
			for (int angle_index = 0; angle_index < 3; angle_index++)
			{
				icp_config.iters_per_layer = iters_per_layer[layer_index];
				icp_config.distance_thresh = distance_thresh[dist_index];
				icp_config.angle_thresh = angle_thresh[angle_index];
				KinectFusion awesome(dataset_config, icp_config);
				awesome.startTracking(790);
			}
		}
	}*/
	
}