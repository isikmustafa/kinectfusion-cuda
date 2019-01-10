#pragma once
#include "glm_macro.h"
#include <glm/mat4x4.hpp>
#include <glm/mat3x3.hpp>
#include <glm/vec3.hpp>

class RigidTransform3D
{
public:
    RigidTransform3D();
    ~RigidTransform3D();

    glm::mat4x4 getHomoMat() const;
    glm::mat3x3 getRotationMat() const;
    glm::vec3 getTranslationVec() const;

    void setHomoMat(glm::mat4x4 mat);
    void setRotationMat(glm::mat3x3 mat);
    void setTranslationVec(glm::vec3 vec);

private:
    glm::mat4x4 m_homo_mat;
};

