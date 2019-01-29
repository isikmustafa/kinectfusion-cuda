// Mostly from https://github.com/isikmustafa/pathtracer/blob/master/camera.cuh

#pragma once

#include <glm/glm.hpp>

class FpsCamera
{
public:
	FpsCamera(const glm::mat4& pose);

	void move(float x_disp, float z_disp);
	void rotate(float radian_world_up, float radian_right);
	glm::mat4 getPose() const;

private:
	glm::mat3 m_rotation;
	glm::vec3 m_position;

private:
	void orthonormalize();
};