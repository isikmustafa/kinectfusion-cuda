#include "sensor.h"

Sensor::Sensor()
	: m_pose(1.0f)
	, m_inverse_pose(1.0f)
	, m_intr(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f))
	, m_inverse_intr(glm::inverse(m_intr))
{}

const glm::mat4& Sensor::getPose() const
{
	return m_pose;
}

const glm::mat4& Sensor::getInversePose() const
{
	return m_inverse_pose;
}

void Sensor::setPose(const glm::mat4& pose)
{
	m_pose = pose;
	m_inverse_pose = glm::inverse(pose);
}

const glm::mat3& Sensor::getIntr() const
{
	return m_intr;
}

const glm::mat3& Sensor::getInverseIntr() const
{
	return m_inverse_intr;
}

glm::vec3 Sensor::getPosition() const
{
	return glm::vec3(m_pose[3]);
}