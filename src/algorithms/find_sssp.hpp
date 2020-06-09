#ifndef ALGORITHMS_FIND_SSSP_HPP
#define ALGORITHMS_FIND_SSSP_HPP

#include "../BidirectedGraphGPU.hpp"

#define DEBUG_SSSP

using distance_t = unsigned long long int;
void find_sssp_gpu(BidirectedGraphGPU& graph, distance_t** distances);

#endif /* ALGORITHMS_FIND_SSSP_HPP */