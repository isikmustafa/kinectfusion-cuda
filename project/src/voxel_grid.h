#pragma once

#include <cuda_runtime.h>
#include <glm/vec3.hpp>
#include <glm/geometric.hpp>

struct Voxel
{
	//The current truncated signed distance value.
	float f{ 0.0f };

	//The current weight;
	float w{ 0.0f };
};

struct VoxelGridStruct
{
	//This is the total width, in meters, of the environment you want to capture.
	//So, this variable decides on voxel resolution together with variable 'n' below.
	const float total_width_in_meters;

	//This defines the number of voxels in the grid.
	//There will be n^3 voxels in the grid.
	const int n;

	//Defines the resolution of a single voxel.
	//One voxel will be of volume resolution^3 where resolution=(total_width_in_meters/n).
	const float resolution;

	//Constant in the paper for TSDF calculations.
	const float mue;

	//cuda_array is the single data source for both surface_object and texture_object.
	//surface_object is used when tsdf since it allows both reading and writing.
	//texture_object is used when raycasting since it allows auto trilinear interpolation.
	cudaArray* cuda_array;
	cudaSurfaceObject_t surface_object;
	cudaTextureObject_t texture_object;

	VoxelGridStruct(float p_total_width_in_meters, int p_n)
		: total_width_in_meters(p_total_width_in_meters)
		, n(p_n)
		, resolution(p_total_width_in_meters / (p_n - 1))
		, mue(2.0f * resolution)
	{}
};

class VoxelGrid
{
public:
	VoxelGrid(float p_total_width_in_meters, int p_n);
	~VoxelGrid();

	const VoxelGridStruct& getStruct() const { return m_struct; }

private:
	VoxelGridStruct m_struct;
};