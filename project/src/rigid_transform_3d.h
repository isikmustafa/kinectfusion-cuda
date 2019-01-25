#pragma once
#include "glm_macro.h"
#include <glm/glm.hpp>

struct RigidTransform3D
{
    glm::mat3x3 rot_mat;
    glm::vec3 transl_vec;

	glm::mat4 getTransformation() const
	{
		glm::mat4x4 pose(rot_mat);
		pose[3] = glm::vec4(transl_vec, 1.0f);

		return pose;
	}
};
