#pragma once

#include "data_helper.h"

#include <array>
#include <type_traits>
#include <glm/glm.hpp>

struct NdtVoxel
{
	glm::vec3 mean{ 0.0f, 0.0f, 0.0f };
	glm::vec3 co_moments_diag{ 0.0f, 0.0f, 0.0f };
	glm::vec3 co_moments_triangle{ 0.0f, 0.0f, 0.0f };
	float count{ 0.0f };
};

template<size_t N>
using VoxelCubeGrid = std::array<std::array<std::array<NdtVoxel, N>, N>, N>;

/*
	Cubic voxel grid representing a 3D NDT map with a voxel_size of N x N x N.
*/
template<size_t N>
class NdtMap3D
{
public:
	NdtMap3D(float total_width_in_meters)
		: m_total_width(total_width_in_meters)
		, m_half_total_width(total_width_in_meters * 0.5f)
		, m_voxel_width(m_total_width / N)
		, m_one_over_voxel_width(1.0f / m_voxel_width)
	{
		static_assert(N % 2 == 0, "Error: Voxel grid dimension must be even.");
	}

private:
	VoxelCubeGrid<N> m_voxel_grid;
	float m_total_width;
	float m_half_total_width;
	float m_voxel_width;
	float m_one_over_voxel_width;

public:
	NdtVoxel& getVoxel(const Coords3D& index)
	{
		return m_voxel_grid[index.x][index.y][index.z];
	}

	float getVoxelWidth() const
	{
		return m_voxel_width;
	}

	/*
		Updates the map given a single measured point, which is expected in world coordinate frame, whose origin is
		at the center of the voxel grid per default.
		Online algorithm explained at https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online.
	*/
	void updateMap(const glm::vec3& point)
	{
		auto grid_coordinate = getGridCoordinate(point);
		auto index = calculateGridIndexFromGridCoordinate(grid_coordinate);

		if (isIndexValid(index))
		{
			auto& voxel = getVoxel(index);
			auto point_local = toVoxelCoordinates(grid_coordinate, index);

			voxel.count += 1.0f;

			// Compute and save p_k - mean_{k-1}
			auto deviations_from_previous_mean = point_local - voxel.mean;

			// Update the mean
			voxel.mean += deviations_from_previous_mean / voxel.count;

			// Update the co-moments
			auto deviations_from_current_mean = point_local - voxel.mean;

			// 1. diagonal elements:
			voxel.co_moments_diag += deviations_from_previous_mean * deviations_from_current_mean;

			// 2. upper triangle elements
			voxel.co_moments_triangle.x += deviations_from_previous_mean.x * deviations_from_current_mean.y;
			voxel.co_moments_triangle.y += deviations_from_previous_mean.x * deviations_from_current_mean.z;
			voxel.co_moments_triangle.z += deviations_from_previous_mean.y * deviations_from_current_mean.z;
		}
	}

	glm::vec3 getGridCoordinate(const glm::vec3& point) const
	{
		return (point + m_half_total_width) * m_one_over_voxel_width;
	}

	Coords3D calculateGridIndexFromGridCoordinate(const glm::vec3& grid_coordinate) const
	{
		return { static_cast<int>(grid_coordinate.x),
				 static_cast<int>(grid_coordinate.y),
				 static_cast<int>(grid_coordinate.z) };
	}

	Coords3D calculateGridIndex(const glm::vec3& point) const
	{
		return calculateGridIndexFromGridCoordinate(getGridCoordinate(point));
	}

	bool isIndexValid(const Coords3D& index) const
	{
		bool x_valid = index.x >= 0 && index.x < N;
		bool y_valid = index.y >= 0 && index.y < N;
		bool z_valid = index.z >= 0 && index.z < N;

		return x_valid && y_valid && z_valid;
	}

	glm::vec3 toVoxelCoordinates(const glm::vec3& grid_coordinate, const Coords3D& index) const
	{
		return grid_coordinate - glm::vec3(index.x, index.y, index.z) - glm::vec3(0.5f);
	}
};