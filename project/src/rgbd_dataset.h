#pragma once

#include <string>
#include <utility>
#include <vector>
#include <glm/glm.hpp>

/*
A class to read RGBD dataset with grountruth camera poses.
https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
*/
class RgbdDataset
{
public:
	void load(const std::string& path);
	std::string getNextDepthImagePath();
    glm::mat4 getCurrentPose();
    glm::mat4 getInitialPoseInverse();
	bool isFinished() const;

private:
    glm::mat4 m_initial_pose_inverse;
	std::vector<std::pair<std::string, glm::mat4>> m_depth_pose_pairs;
	int m_current_index;
};