#include "rigid_transform_3d.h"

RigidTransform3D::RigidTransform3D()
{
    m_homo_mat = glm::mat4(1.0);
}

RigidTransform3D::~RigidTransform3D()
{
}

glm::mat4x4 RigidTransform3D::getHomoMat() const
{
    return m_homo_mat;
}

glm::mat3x3 RigidTransform3D::getRotationMat() const
{
    return glm::mat3(glm::vec3(m_homo_mat[0]), glm::vec3(m_homo_mat[1]), glm::vec3(m_homo_mat[2]));
}

glm::vec3 RigidTransform3D::getTranslationVec() const
{
    return glm::vec3(m_homo_mat[3]);
}

void RigidTransform3D::setHomoMat(glm::mat4x4 mat)
{
    m_homo_mat = mat;
}

void RigidTransform3D::setRotationMat(glm::mat3x3 mat)
{
    for (int col = 0; col < 3; col++)
    {
        for (int row = 0; row < 3; row++)
        {
            m_homo_mat[col][row] = mat[col][row];
        }
    }
}

void RigidTransform3D::setTranslationVec(glm::vec3 vec)
{
    for (int row = 0; row < 3; row++)
    {
        m_homo_mat[3][row] = vec[row];
    }
}
