#pragma once
#include "pch.h"

#include "cuda_grid_map.h"

glm::vec2 computeCorrespondenceTestWrapper(glm::vec3 vertex, glm::mat3 rotation_mat, glm::vec3 translation_vec, 
    glm::mat3 intrinsics);

glm::vec3 computeNormalTestWrapper(CudaGridMap &vertex_map, unsigned int u, unsigned int v);

bool normalsAreTooDifferentTestWrapper(glm::vec3 normal, glm::vec3 target_normal, glm::mat3 rotation_mat,
    float angle_thresh);

void computeAndFillATestWrapper(std::array<float, 6> *result, glm::vec3 vertex, glm::vec3 normal);

float computeAndFillBTestWrapper(glm::vec3 vertex, glm::vec3 target_vertex, glm::vec3 target_normal);