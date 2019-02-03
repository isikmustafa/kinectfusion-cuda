#pragma once

#include <string>
#include <array>

#include "cuda_grid_map.h"
#include "grid_map_pyramid.h"

std::pair<float,float> poseError(glm::mat4 pose_1, glm::mat4 pose_2);
void writeSurface1x32(const std::string& file_name, cudaArray* gpu_source, int width, int height);
void writeSurface4x32(const std::string& file_name, cudaArray* gpu_source, int width, int height);
void writeDepthPyramidToFile(const std::string& file_name, GridMapPyramid<CudaGridMap> pyramid);
void writeVectorPyramidToFile(const std::string& file_name, std::array<CudaGridMap, 3> pyramid);
