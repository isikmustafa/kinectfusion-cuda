// Mostly from https://github.com/isikmustafa/pathtracer/blob/master/camera.cu

#include "fps_camera.h"

#include <glm/gtc/quaternion.hpp>

FpsCamera::FpsCamera(const glm::mat4& pose)
	: m_rotation(pose)
	, m_position(pose[3])
{}

void FpsCamera::move(float x_disp, float z_disp)
{
	m_position += m_rotation[0] * x_disp;
	m_position += m_rotation[2] * z_disp;
}

void FpsCamera::rotate(float radian_world_up, float radian_right)
{
	auto rot_on_y = glm::angleAxis(radian_world_up, glm::vec3(0.0f, 1.0f, 0.0f));
	auto rot_on_right = glm::angleAxis(radian_right, m_rotation[0]);

	m_rotation[0] = rot_on_y * glm::vec3(m_rotation[0]);
	m_rotation[1] = rot_on_y * rot_on_right * glm::vec3(m_rotation[1]);
	m_rotation[2] = rot_on_y * rot_on_right * glm::vec3(m_rotation[2]);

	orthonormalize();
}

glm::mat4 FpsCamera::getPose() const
{
	glm::mat4 pose(m_rotation);
	pose[3] = glm::vec4(m_position, 1.0f);

	return pose;
}

void FpsCamera::orthonormalize()
{
	m_rotation[2] = glm::normalize(m_rotation[2]);
	m_rotation[0] = glm::normalize(glm::cross(m_rotation[1], m_rotation[2]));
	m_rotation[1] = glm::normalize(glm::cross(m_rotation[2], m_rotation[0]));
}