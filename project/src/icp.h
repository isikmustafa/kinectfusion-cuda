#pragma once
#include <vector>

#include "cuda_grid_map.h"
#include "grid_map_pyramid.h"
#include "rigid_transform_3d.h"

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
    ICP(RigidTransform3D previous_transform, std::vector<unsigned int> iters_per_layer, unsigned int width, 
        unsigned int height, float distance_thresh, float angle_thresh);
    ~ICP();

    RigidTransform3D computePose(GridMapPyramid<CudaGridMap> &vertex_pyramid, 
        GridMapPyramid<CudaGridMap> &target_vertex_pyramid, RigidTransform3D &previous_pose);

private:
    // Buffer for the target normals, allocated once
    GridMapPyramid<CudaGridMap> m_target_normal_pyramid;
    
    // Buffers for the residual paramterers, allocated once
    float *m_mat_a[6];
    float *m_vec_b;

    cudaChannelFormatDesc m_normal_format_description;
    RigidTransform3D m_previous_transform;
    std::vector<unsigned int> m_iters_per_layer;
    float m_distance_thresh;
    float m_angle_thresh;
};

