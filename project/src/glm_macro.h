#pragma once
// ALWAYS INCLUDE BEFORE ANY GLM INCLUDES

// This flag is needed to avoid strange error in glm header (appeared after moving the intrinsics matrix to this file)
#define GLM_FORCE_PURE

// This flag allows to use experimental features of glm
#define GLM_ENABLE_EXPERIMENTAL

#ifdef _DEBUG
#define GLM_FORCE_CUDA
#endif