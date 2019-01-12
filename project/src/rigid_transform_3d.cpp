#pragma once
#include "rigid_transform_3d.h"

#include <cuda_runtime.h>
#include "cuda_utils.h"

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
    glm::mat4x4 *new_mat;
    cudaMalloc((void**)&new_mat, sizeof(glm::mat4x4));
    cudaMemcpy(new_mat, m_homo_mat, sizeof(glm::mat4x4), cudaMemcpyDeviceToDevice);
    return new_mat;
}

void RigidTransform3D::setHomoMat(glm::mat4x4 *mat)
{
    if (isDeviceMemory(mat))
    {
        cudaMemcpy(m_homo_mat, mat, sizeof(glm::mat4x4), cudaMemcpyDeviceToDevice);
    }
    else
    {
        cudaMemcpy(m_homo_mat, mat, sizeof(glm::mat4x4), cudaMemcpyHostToDevice);
    }
}
