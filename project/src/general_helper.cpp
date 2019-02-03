#include "general_helper.h"

#undef STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/quaternion.hpp>
#include <glm/gtx/vector_angle.hpp>

#include "cuda_utils.h"

std::pair <float, float> poseError(glm::mat4 pose_1, glm::mat4 pose_2)
{
	glm::vec3 v = glm::normalize(glm::vec3(1.0, 1.0, 1.0));
	glm::vec3 true_rotated_v = glm::mat3(pose_1) * v;
	glm::vec3 rotated_v = glm::mat3(pose_2) * v;
	float angle_error = glm::distance(true_rotated_v, rotated_v);

	float distance_error = glm::distance(glm::vec3(pose_1[3]), glm::vec3(pose_2[3]));

    return std::pair <float, float>(angle_error, distance_error);
}

void writeSurface1x32(const std::string& file_name, cudaArray* gpu_source, int width, int height)
{
    std::unique_ptr<float[]> float_data(new float[width * height]);
    auto float_data_ptr = float_data.get();

    HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4, cudaMemcpyDeviceToHost));

    std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height]);
    auto byte_data_ptr = byte_data.get();

    int size = width * height;
    for (int i = 0; i < size; ++i)
    {
        byte_data_ptr[i] = static_cast<unsigned char>(float_data_ptr[i] * 64.0f);
    }

    auto final_path = std::to_string(width) + "x" + std::to_string(height) + file_name + ".png";
    stbi_write_png(final_path.c_str(), width, height, 1, byte_data_ptr, width);
}

void writeSurface4x32(const std::string& file_name, cudaArray* gpu_source, int width, int height)
{
    std::unique_ptr<float[]> float_data(new float[width * height * 4]);
    auto float_data_ptr = float_data.get();

    HANDLE_ERROR(cudaMemcpyFromArray(float_data_ptr, gpu_source, 0, 0, width * height * 4 * 4, cudaMemcpyDeviceToHost));

    std::unique_ptr<unsigned char[]> byte_data(new unsigned char[width * height * 3]);
    auto byte_data_ptr = byte_data.get();

    int size = width * height;
    for (int i = 0; i < size; ++i)
    {
        int idx_byte = i * 3;
        int idx_float = i * 4;
        byte_data_ptr[idx_byte] = static_cast<unsigned char>(float_data_ptr[idx_float] * 255.0f);
        byte_data_ptr[idx_byte + 1] = static_cast<unsigned char>(float_data_ptr[idx_float + 1] * 255.0f);
        byte_data_ptr[idx_byte + 2] = static_cast<unsigned char>(float_data_ptr[idx_float + 2] * 255.0f);
    }

	auto final_path = std::to_string(width) + "x" + std::to_string(height) + file_name + ".png";
    stbi_write_png(final_path.c_str(), width, height, 3, byte_data_ptr, width * 3);
}

void writeDepthPyramidToFile(const std::string& file_name, GridMapPyramid<CudaGridMap> pyramid)
{
    for (int i = 0; i < 3; i++)
    {
        auto dims = pyramid[i].getGridDims();
        writeSurface1x32(file_name, pyramid[i].getCudaArray(), dims[0], dims[1]);
    }
}

void writeVectorPyramidToFile(const std::string& file_name, std::array<CudaGridMap, 3> pyramid)
{
    for (const CudaGridMap &map : pyramid)
    {
        auto dims = map.getGridDims();
        writeSurface4x32(file_name, map.getCudaArray(), dims[0], dims[1]);
    }
}