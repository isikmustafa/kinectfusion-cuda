#pragma once

#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

struct Voxel
{
	//The current truncated signed distance value.
	float f;

	//The current weight;
	float w;
};

struct VoxelGridStruct
{
	//This is the total width, in millimeters, of the environment you want to capture.
	//So, this variable decides on voxel resolution together with variable 'n' below.
	float total_width_in_millimeters;

	//This defines the number of voxels in the grid.
	//There will be n^3 voxels in the grid.
	//One voxel will be of volume resolution^3 where resolution=(total_width_in_meters/n)
	int n;

	//Pointer to allocated memory on GPU for voxels.
	cudaPitchedPtr pointer;

	VoxelGridStruct(float p_total_width_in_millimeters, int p_n)
		: total_width_in_millimeters(p_total_width_in_millimeters)
		, n(p_n)
	{}

	//Checks if the point (world coordinate) is inside the voxel.
	__device__ bool isPointIn(const glm::vec3& point) const
	{
		auto half_total_width = total_width_in_millimeters * 0.5f;
		auto abs_point = glm::abs(point);

		return abs_point.x < half_total_width && abs_point.y < half_total_width && abs_point.z < half_total_width;
	}
};

class VoxelGrid
{
public:
	VoxelGrid(float p_total_width_in_millimeters, int p_n);
	~VoxelGrid();

	const VoxelGridStruct& getStruct() const { return m_struct; }

private:
	VoxelGridStruct m_struct;
};