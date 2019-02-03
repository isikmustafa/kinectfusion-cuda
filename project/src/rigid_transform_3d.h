#pragma once
#include "glm_macro.h"
#include <glm/glm.hpp>

struct RigidTransform3D
{
	glm::mat3 rot_mat;
	glm::vec3 transl_vec;

	RigidTransform3D() = default;
	RigidTransform3D(const glm::mat4& transformation)
		: rot_mat(transformation)
		, transl_vec(transformation[3])
	{}

	glm::mat4 getTransformation() const
	{
		glm::mat4 pose(rot_mat);
		pose[3] = glm::vec4(transl_vec, 1.0f);

		return pose;
	}
};
