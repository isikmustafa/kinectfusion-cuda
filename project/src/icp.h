#pragma once
#include <vector>

#include "cuda_grid_map.h"
#include "grid_map_pyramid.h"
#include "rigid_transform_3d.h"
#include "linear_least_squares.h"
#include "sensor.h"

struct IcpConfig
{
    std::vector<unsigned int> iters_per_layer;
    unsigned int width;
    unsigned int height;
    float distance_thresh;
    float angle_thresh; 
	float iteration_stop_thresh_angle;
	float iteration_stop_thresh_distance;
};

/*
    Class for a specific implementation of the ICP algorithm. It computes the global pose of a set of vertices by 
    aligning it to a target set of vertices. It uses a coarse to fine approach by starting from
    a lower resolution which is successively increased over the iterations. On creation only some general configuration
    parameters are specified. After instantiation, the object can be used to compute poses arbitrarily often, as long as 
    the dimensions match.
*/
class ICP
{
public:
    ICP(const IcpConfig& config);
    ~ICP();

    RigidTransform3D computePose(GridMapPyramid<CudaGridMap>& vertex_pyramid,
        GridMapPyramid<CudaGridMap>& target_vertex_pyramid, GridMapPyramid<CudaGridMap>& target_normal_pyramid, const Sensor& sensor);
    
    std::array<float, 2> getExecutionTimes();

private:
	IcpConfig m_icp_config;
    std::array<float, 2> m_execution_times;
    
    LinearLeastSquares solver;

    // Buffers for the residual paramterers and the result, allocated once
    float* m_mat_a;
    float* m_vec_b;
    float* m_vec_x;

private:
    void updatePose(RigidTransform3D& pose);
    glm::mat3x3 buildRotationZYX(float z_angle, float y_angle, float x_angle);
    unsigned int countResiduals(unsigned int max_idx);
};