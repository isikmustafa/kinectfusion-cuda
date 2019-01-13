#pragma once
#include <vector>

#include "glm_macro.h"
#include <glm/glm.hpp>
#include "cuda_grid_map.h"
#include "depth_map.h"
#include "grid_map_pyramid.h"
#include "rigid_transform_3d.h"
#include "validity_mask.h"

/*
    Class for a specific implementation of the ICP algorithm. It computes the global pose of a set of vertices by 
    aligning it to a target set of vertices. It uses a coarse to fine approach by starting from
    a lower resolution which is successively increased over the iterations. On creation only the dimensions of the 
    underlying grid and of the resolution pyramids are specified. After instantiation, the object can be used to 
    compute poses arbitrarily often, as long as the dimensions match.
*/
class ICP
{
public:
    ICP(unsigned int grid_width, unsigned int grid_height, cudaChannelFormatDesc format_description,
        std::vector<unsigned int> iters_per_layer, float distance_thresh, float angle_thresh);
    ~ICP();

    RigidTransform3D computePose(
        GridMapPyramid<CudaGridMap> &vertex_pyramid, GridMapPyramid<CudaGridMap> &target_vertex_pyramid, 
        RigidTransform3D &previous_pose, ValidityMask validity_mask);

private:
    std::vector<unsigned int> m_iters_per_layer;
    float m_distance_thresh;
    float m_angle_thresh;

    // These grid maps serve as buffers for intermediate results to avoid repeated reallocations of memory.
    GridMapPyramid<CudaGridMap> m_transformed_vertex_pyramid;
    GridMapPyramid<CudaGridMap> m_normal_pyramid;
    GridMapPyramid<CudaGridMap> m_target_normal_pyramid;
};

