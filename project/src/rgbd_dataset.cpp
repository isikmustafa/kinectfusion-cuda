#include "rgbd_dataset.h"

#include <fstream>
#include <glm/gtc/quaternion.hpp>

void RgbdDataset::load(const std::string& path)
{
	//Reset.
	m_depth_pose_pairs.clear();
	m_current_index = 0;

	std::vector<double> timestamps;
	std::vector<glm::vec3> translations;
	std::vector<glm::tquat<float>> quats;
	std::ifstream groundtruth_file(path + "/groundtruth.txt");

	//Get rid of first three info lines.
	std::string temp;
	std::getline(groundtruth_file, temp);
	std::getline(groundtruth_file, temp);
	std::getline(groundtruth_file, temp);

	double timestamp;
	glm::vec3 translation;
	glm::tquat<float> quat;
	while (groundtruth_file >> timestamp)
	{
		timestamps.push_back(timestamp);

		groundtruth_file >> translation.x >> translation.y >> translation.z;
		translations.push_back(translation);

		groundtruth_file >> quat.x >> quat.y >> quat.z >> quat.w;
		quats.push_back(quat);
	}

	std::ifstream depth_file(path + "/depth.txt");

	//Get rid of first three info lines.
	std::getline(depth_file, temp);
	std::getline(depth_file, temp);
	std::getline(depth_file, temp);

	int last_index = 0;
	int ts_size = timestamps.size();
	std::string depth_image_path;
	while ((depth_file >> timestamp))
	{
		//Search for closest possible sensor pose in the groundtruth data.
		double min = std::numeric_limits<double>::max();
		int min_index = -1;
		for (int i = last_index; i < ts_size; ++i)
		{
			auto diff = std::abs(timestamp - timestamps[i]);
			if (diff < min)
			{
				min = diff;
				min_index = i;
			}
			else
			{
				last_index = min_index;
				break;
			}
		}

		depth_file >> depth_image_path;

		//Construct the pose.
		auto pose = glm::mat4_cast(quats[min_index]);
		pose[3] = glm::vec4(translations[min_index], 1.0f);

		m_depth_pose_pairs.emplace_back(path + "/" + depth_image_path, pose);
	}
}

std::pair<std::string, glm::mat4> RgbdDataset::nextDepthAndPose()
{
	return m_depth_pose_pairs[m_current_index++];
}

std::string RgbdDataset::nextDepth()
{
    return nextDepthAndPose().first;
}

bool RgbdDataset::isFinished() const
{
	return m_current_index >= m_depth_pose_pairs.size();
}
