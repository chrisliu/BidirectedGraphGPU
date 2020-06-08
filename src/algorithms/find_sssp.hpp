#ifndef ALGORITHMS_FIND_SSSP_HPP
#define ALGORITHMS_FIND_SSSP_HPP

#include "../BidirectedGraphGPU.hpp"

void find_sssp_gpu(BidirectedGraphGPU& graph, nid_t* distances);

#endif /* ALGORITHMS_FIND_SSSP_HPP */