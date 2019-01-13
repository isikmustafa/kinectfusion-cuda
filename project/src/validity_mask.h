#pragma once
#include <vector>

#include "cuda_grid_map.h"

/*
    This special grid map can be used for managing efficient and convenient validity masks. At the beginning, all 
    elements are valid per default. A validity mask maintains an index that lists the valid elemets in an order. 
    The index is not kept coherrent to the true mask (grid map) automatically and needs manual updates.
*/
class ValidityMask : public CudaGridMap
{
public:
    ValidityMask(unsigned int width, unsigned int height, cudaChannelFormatDesc channel_description); 
    ~ValidityMask();

    // The copy constructor doesn't ensure coherence between index and grid map
    ValidityMask(const ValidityMask &validity_mask);

    unsigned int getIndexLength();
    bool isValid(unsigned int u, unsigned int v);

    // This function can be used to iterate only over valid elements or to enable a more efficient data distribution 
    // in CUDA kernels.
    std::array<unsigned int, 2> getValidCoordinates(unsigned int i);
    
    void invalidate(unsigned int u, unsigned v);
    void updateIndex();
    
private:
    unsigned int m_n_valid;
    std::vector<std::array<unsigned int, 2>> m_index;
};

