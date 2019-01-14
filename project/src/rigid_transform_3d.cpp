#pragma once
#include "rigid_transform_3d.h"

#include <cuda_runtime.h>
#include "cuda_utils.h"

RigidTransform3D::RigidTransform3D()
{ 
    glm::mat4x4 homo_mat = glm::mat4(1.0);
}

RigidTransform3D::~RigidTransform3D()
{
}

glm::mat4x4 RigidTransform3D::getHomoMat()
{
    return m_homo_mat;
}

void RigidTransform3D::setHomoMat(glm::mat4x4 &mat)
{
    m_homo_mat = mat;
}
