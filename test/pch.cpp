//
// pch.cpp
// Include all .cpp from the main project that are used in the tests here.
//

#include "pch.h"

#include "cuda_grid_map.h"
#include "depth_map.h"
#include "rigid_transform_3d.h"
#include "icp.h"
#include "cuda_event.h"
#include "general_helper.h"
#include "linear_least_squares.h"
#include "rgbd_dataset.h"

#include "cuda_grid_map.cpp"
#include "depth_map.cpp"
#include "icp.cpp"
#include "cuda_event.cpp"
#include "general_helper.cpp"
#include "linear_least_squares.cpp"
#include "rgbd_dataset.cpp"