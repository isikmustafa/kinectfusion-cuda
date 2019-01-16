#pragma once

#include <glm/mat3x3.hpp>
#include <glm/mat4x4.hpp>

class Sensor
{
public:
	Sensor();

	const glm::mat4& getPose() const;
	const glm::mat4& getInversePose() const;
	void setPose(const glm::mat4& pose);
	const glm::mat3& getIntr() const;
	const glm::mat3& getInverseIntr() const;
	glm::vec3 getPosition() const;

private:
	glm::mat4 m_pose;
	glm::mat4 m_inverse_pose;
	const glm::mat3 m_intr;
	const glm::mat3 m_inverse_intr;
};