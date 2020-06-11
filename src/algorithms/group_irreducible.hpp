#ifndef GROUP_IRREDUCIBLE_HPP
#define GROUP_IRREDUCIBLE_HPP

#include "../BidirectedGraphGPU.hpp"

#define DEBUG_GROUP_IRREDUCIBLE

void group_irreducible(BidirectedGraphGPU& graph, nid_t *reducible, size_t count, nid_t **membership);

#endif /* GROUP_IRREDUCIBLE_HPP */