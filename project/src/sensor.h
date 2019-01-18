#pragma once

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

class Sensor
{
public:
	Sensor()
		: m_pose(1.0f)
		, m_inverse_pose(1.0f)
		, m_intr(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f))
		, m_inverse_intr(glm::inverse(m_intr))
	{}

	__host__ __device__ const glm::mat4& getPose() const { return m_pose; }
	__host__ __device__ const glm::mat4& getInversePose() const { return m_inverse_pose; }
	__host__ __device__ void setPose(const glm::mat4& pose) { m_pose = pose; m_inverse_pose = glm::inverse(pose); }
	__host__ __device__ const glm::mat3& getIntr() const { return m_intr; }
	__host__ __device__ const glm::mat3& getInverseIntr() const { return m_inverse_intr; }
	__host__ __device__ glm::vec3 getPosition() const { return glm::vec3(m_pose[3]); }

private:
	glm::mat4 m_pose;
	glm::mat4 m_inverse_pose;
	const glm::mat3 m_intr;
	const glm::mat3 m_inverse_intr;
};