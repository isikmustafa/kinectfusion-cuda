#pragma once
#include <array>
#include <type_traits>
#include <glm/glm.hpp>

#include "data_helper.h"

struct NdtVoxel 
{
    glm::fvec3 mean = glm::fvec3(0.0);
    glm::fvec3 co_moments_diag = glm::fvec3(0.0);
    glm::fvec3 co_moments_triangle = glm::fvec3(0.0);
    float count = 0.0f;
};


template<size_t N>
using VoxelCube = std::array<std::array<std::array<NdtVoxel, N>, N>, N>;

/*
    Cubic voxel grid representing a 3D NDT map with a voxel_size of N x N x N.
*/
template<size_t N>
class NdtMap3D
{
public:
    NdtMap3D(float voxel_size) : m_voxel_size(voxel_size)
    { 
        static_assert(N % 2 == 0, "Error: Voxel grid dimension must be even.");
        float center_distance = N / 2 * m_voxel_size;
        m_grid_center = glm::vec3(center_distance);
    }
    ~NdtMap3D() {}

private:
    float m_voxel_size;
    glm::fvec3 m_grid_center;
    VoxelCube<N> m_voxel_grid;

public:
    NdtVoxel& getVoxel(Coords3D coords)
    {
        return m_voxel_grid[coords.x][coords.y][coords.z];
    }

    float getVoxelSize() const
    {
        return m_voxel_size;
    }

    /* 
        Updates the map given a single measured point, which is expected in world coordinate frame, whose origin is 
        at the center of the voxel grid per default.
        Online algorithm explained at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.
    */
    void updateMap(glm::fvec3 &point)
    {
        Coords3D coords = calcCoordinates3D(point);

        if (coordinatesAreValid(coords))
        {
            NdtVoxel &voxel = getVoxel(coords);
            glm::fvec3 point_local = toVoxelCoordinates(point, coords);

            voxel.count += 1.0f;

            // Compute and save p_k - mean_{k-1}
            glm::fvec3 deviations_from_previous_mean = point_local - voxel.mean;

            // Update the mean
            voxel.mean += deviations_from_previous_mean / voxel.count;

            // Update the co-moments
            glm::fvec3 deviations_from_current_mean = point_local - voxel.mean;

            // 1. diagonal elements:
            voxel.co_moments_diag += deviations_from_previous_mean * deviations_from_current_mean;

            // 2. upper triangle elements
            voxel.co_moments_triangle.x += deviations_from_previous_mean.x * deviations_from_current_mean.y;
            voxel.co_moments_triangle.y += deviations_from_previous_mean.x * deviations_from_current_mean.z;
            voxel.co_moments_triangle.z += deviations_from_previous_mean.y * deviations_from_current_mean.z;
        }
    }

    Coords3D calcCoordinates3D(glm::fvec3 &point)
    {
        glm::fvec3 point_in_grid_frame = point + m_grid_center;
        return { (int)std::floor(point_in_grid_frame.x / m_voxel_size),
                 (int)std::floor(point_in_grid_frame.y / m_voxel_size),
                 (int)std::floor(point_in_grid_frame.z / m_voxel_size) };
    }

    glm::fvec3 toVoxelCoordinates(glm::fvec3 &point, Coords3D coords)
    {
        glm::fvec3 voxel_center = m_voxel_size * (glm::fvec3)coords + m_voxel_size / 2.0f;
        return point + m_grid_center - voxel_center;
    }

    bool coordinatesAreValid(Coords3D coords)
    {
        bool x_valid = coords.x >= 0 && coords.x < N;
        bool y_valid = coords.y >= 0 && coords.y < N;
        bool z_valid = coords.z >= 0 && coords.z < N;

        return x_valid && y_valid && z_valid;
    }
};

