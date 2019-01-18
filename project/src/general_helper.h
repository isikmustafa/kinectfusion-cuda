#pragma once
#include <string>
#include <array>
#include "cuda_grid_map.h"

/////////////// DEBUG FUNCTIONS /////////////////////////
void writeSurface1x32(std::string file_name, cudaArray* gpu_source, int width, int height);
void writeSurface4x32(std::string file_name, cudaArray* gpu_source, int width, int height);
void writeDepthPyramidToFile(std::string file_name, std::array<CudaGridMap, 3> pyramid);
void writeVectorPyramidToFile(std::string file_name, std::array<CudaGridMap, 3> pyramid);
