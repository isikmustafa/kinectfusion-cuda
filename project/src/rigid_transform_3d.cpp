#include "rigid_transform_3d.h"

#include <cuda_runtime.h>

RigidTransform3D::RigidTransform3D()
{ 
    glm::mat4x4 homo_mat = glm::mat4(1.0);
    cudaMalloc((void**) &m_homo_mat, sizeof(glm::mat4x4));
    cudaMemcpy(m_homo_mat, &homo_mat, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
}

RigidTransform3D::~RigidTransform3D()
{
    cudaFree(m_homo_mat);
}

glm::mat4x4 *RigidTransform3D::getHomoMat()
{
    return m_homo_mat;
}

void RigidTransform3D::setHomoMat(glm::mat4x4 mat)
{
    cudaMemcpy(m_homo_mat, &mat, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
}
