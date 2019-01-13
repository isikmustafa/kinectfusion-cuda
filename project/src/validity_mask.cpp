#include "validity_mask.h"

ValidityMask::ValidityMask(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description)
    : CudaGridMap(width, height, channel_description)
{};

ValidityMask::~ValidityMask()
{
}
