#pragma once
#include "ndt_map_3d.h"
#include "data_helper.h"

template<size_t N>
class NdtPoseEstimator
{
public:
    NdtPoseEstimator(unsigned int frame_width, unsigned int frame_height, float voxel_size)
        : m_frame_width(frame_width), m_frame_height(frame_height), m_ndt_map(voxel_size) {}
    ~NdtPoseEstimator() {}

    void initialize(float *depth_data)
    {
        // TODO: implement
    }

private:
    unsigned int m_frame_width;
    unsigned int m_frame_height;
    NdtMap3D<N> m_ndt_map;

private:
    Coords2D calcPixelCoords(unsigned int idx)
    {
        return { idx / m_frame_width, idx % m_frame_width };
    }
    
    glm::vec3 to3dPoint(float depth, Coords2D coords)
    {
        // TODO: implement
    }
};