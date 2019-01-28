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

    // Vertices always expected in world coordinates
    void initialize(glm::fvec4 *vertices)
    {
        Coords2D coords;
        for (coords.x = 0; coords.x < m_frame_height; coords.x++)
        {
            for (coords.y = 0; coords.y < m_frame_width; coords.y++)
            {
                glm::fvec4 vertex = vertices[calcIndexFromPixelCoords(coords)];

                m_ndt_map.updateMap(glm::fvec3(vertex));
            }
        }
    }

    // Vertices always expected in world coordinates
    void computePose(glm::fvec4 *vertices, glm::mat4x4 previous_pose)
    {
        /*
            For each vertex:
            1. Calculate grid coordinates
            2. Get normal distribution of corresponding voxel
            3. Optimize pose with Newton, following chapter V of
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.10.7059&rep=rep1&type=pdf
            4. return if converged, else back to 1
        */
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

    int calcIndexFromPixelCoords(Coords2D coords)
    {
        return coords.x * m_frame_width + coords.y;
    }
};