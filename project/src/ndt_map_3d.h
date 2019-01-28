#pragma once
#include <array>
#include <glm/glm.hpp>

struct Coords3D
{
    unsigned int x;
    unsigned int y;
    unsigned int z;
};

struct NdtVoxel 
{
    glm::vec3 mean = glm::fvec3(0.0);
    glm::vec3 co_moments_diag = glm::fvec3(0.0);
    glm::vec3 co_moments_triangle = glm::fvec3(0.0);
    float count = 0.0f;
};

template<size_t dim>
using VoxelCube = std::array<std::array<std::array<NdtVoxel, dim>, dim>, dim>;

template<size_t dim>
class NdtMap3d
{
public:
    NdtMap3d() {}
    ~NdtMap3d() {}

private:
    VoxelCube<dim> m_voxel_grid;

public:
    NdtVoxel& getVoxel(Coords3D coordinates)
    {
        return m_voxel_grid[coordinates.x][coordinates.y][coordinates.z];
    }

    // Algorithm explained at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online
    void updateVoxel(glm::vec3 point, Coords3D coordinates)
    {
        NdtVoxel &voxel = getVoxel(coordinates);
        voxel.count += 1.0f;
        
        // Compute and save p_k - mean_{k-1}
        glm::vec3 deviations_from_previous_mean = point - voxel.mean;

        // Update the mean
        voxel.mean += deviations_from_previous_mean / voxel.count;

        // Update the co-moments
        glm::vec3 deviations_from_current_mean = point - voxel.mean;

        // 1. diagonal elements:
        voxel.co_moments_diag += deviations_from_previous_mean * deviations_from_current_mean;

        // 2. upper triangle elements
        voxel.co_moments_triangle.x += deviations_from_previous_mean.x * deviations_from_current_mean.y;
        voxel.co_moments_triangle.y += deviations_from_previous_mean.x * deviations_from_current_mean.z;
        voxel.co_moments_triangle.z += deviations_from_previous_mean.y * deviations_from_current_mean.z;
    }
};

