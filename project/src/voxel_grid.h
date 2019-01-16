#pragma once

#include <cuda_runtime.h>

struct Voxel
{
	//The current truncated signed distance value.
	float f;

	//The current weight;
	float w;
};

struct VoxelGrid
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

	VoxelGrid(float p_total_width_in_millimeters, int p_n);
	~VoxelGrid();
};