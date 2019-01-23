#pragma once

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

class Sensor
{
public:
	Sensor()
		: m_pose(1.0f)
		, m_inverse_pose(1.0f)
	{
		constexpr auto fovx = 525.0f;
		constexpr auto fovy = 525.0f;

		m_intr[0] = glm::mat3(glm::vec3(fovx, 0.0f, 0.0f), glm::vec3(0.0f, fovy, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
		m_intr[1] = glm::mat3(glm::vec3(fovx / 2, 0.0f, 0.0f), glm::vec3(0.0f, fovy / 2, 0.0f), glm::vec3(159.5f, 119.5f, 1.0f));
		m_intr[2] = glm::mat3(glm::vec3(fovx / 4, 0.0f, 0.0f), glm::vec3(0.0f, fovy / 4, 0.0f), glm::vec3(79.5f, 59.5f, 1.0f));
		m_inverse_intr[0] = glm::inverse(m_intr[0]);
		m_inverse_intr[1] = glm::inverse(m_intr[1]);
		m_inverse_intr[2] = glm::inverse(m_intr[2]);
	}

	__host__ __device__ const glm::mat4& getPose() const
	{
		return m_pose;
	}
	__host__ __device__ const glm::mat4& getInversePose() const
	{
		return m_inverse_pose;
	}
	__host__ __device__ void setPose(const glm::mat4& pose)
	{
		m_pose = pose;
		m_inverse_pose = glm::inverse(pose);
	}
	__host__ __device__ glm::mat3 getIntr(int downsampling_level) const
	{
		return m_intr[downsampling_level];
	}
	__host__ __device__ glm::mat3 getInverseIntr(int downsampling_level) const
	{
		return m_inverse_intr[downsampling_level];
	}
	__host__ __device__ glm::vec3 getPosition() const
	{
		return glm::vec3(m_pose[3]);
	}

private:
	glm::mat4 m_pose;
	glm::mat4 m_inverse_pose;
	glm::mat3 m_intr[3];
	glm::mat3 m_inverse_intr[3];
};