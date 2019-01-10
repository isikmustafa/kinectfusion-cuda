#pragma once
#include "glm_macro.h"
#include <cuda_runtime.h>
#include <glm/mat3x3.hpp>

/*
    Helper struct to bundle a cuda surface object and a cuda array pointer that belong to each other
*/
struct CudaSurface
{
    cudaSurfaceObject_t surface_object{ 0 };
    cudaArray* cuda_array{ nullptr };
};

/*
    Singleton class containing the sensor intrinsic matrix and its inverse.
*/
class SensorIntrinsics
{
public:
    static glm::mat3 getMat()
    {
        SensorIntrinsics& instance = getInstance();
        return instance.mat;
    }
    static glm::mat3 getInvMat()
    {
        SensorIntrinsics& instance = getInstance();
        return instance.inv_mat;
    }

private:
    SensorIntrinsics() 
    {
        mat = glm::mat3(glm::vec3(525.0f, 0.0f, 0.0f), glm::vec3(0.0f, 525.0f, 0.0f), glm::vec3(319.5f, 239.5f, 1.0f));
        inv_mat = glm::inverse(mat);
    }
    static SensorIntrinsics& getInstance()
    {
        static SensorIntrinsics instance;
        return instance;
    }
    glm::mat3 mat;
    glm::mat3 inv_mat;

public:
    SensorIntrinsics(SensorIntrinsics const&) = delete;
    void operator=(SensorIntrinsics const&) = delete;
};