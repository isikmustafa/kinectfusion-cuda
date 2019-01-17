#pragma once
#include "pch.h"

void computeCorrespondenceTestWrapper(std::array<int, 2> *result, CudaGridMap &vertex_map, glm::mat3x3 &rotation_mat,
    glm::vec3 &translation_vec, glm::mat3x3 &intrinsics);

glm::vec3 computeNormalTestWrapper(CudaGridMap &vertex_map, unsigned int u, unsigned int v);