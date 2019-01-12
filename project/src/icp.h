#pragma once

#include "glm_macro.h"
#include <glm/glm.hpp>
#include "cuda_grid_map.h"
#include "grid_map_pyramid.h"
#include "depth_map.h"
#include "rigid_transform_3d.h"

class ICP
{
public:
    ICP(GridMapPyramid<CudaGridMap> &vertex_pyramid, GridMapPyramid<DepthMap> &depth_predicton_pyramid, RigidTransform3D &previous_pose,
        int n_iterations, float distance_thresh, float angle_thresh);
    ~ICP();

    void initializePose(glm::mat4x4 initial_pose);
    RigidTransform3D computePose();

private:
    int m_n_iterations;
    float m_distance_thresh;
    float m_angle_thresh;

    GridMapPyramid<CudaGridMap>* m_vertex_pyramid;
    GridMapPyramid<CudaGridMap>* m_normal_pyramid;
    GridMapPyramid<DepthMap>* m_depth_prediction_pyramid;
    
    RigidTransform3D m_pose;
    RigidTransform3D m_previous_pose;
};

