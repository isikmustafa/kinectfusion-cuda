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
	std::pair<std::string, glm::mat4> nextDepthAndPose();

private:
	std::vector<std::pair<std::string, glm::mat4>> m_depth_pose_pairs;
	int m_current_index;
};