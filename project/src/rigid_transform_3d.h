#pragma once
#include "glm_macro.h"
#include <glm/glm.hpp>

struct RigidTransform3D
{
    glm::mat3x3 rot_mat;
    glm::vec3 transl_vec;
};

