#ifndef BIDIRECTEDGRAPHGPU_HPP
#define BIDIRECTEDGRAPHGPU_HPP

#include "handle.hpp"

#define DEBUG_HOST

/** Each node (n_l, n_r) of node id nid will have each node side mapped to
 * index of n_l = 2 * nid
 * index of n_r = 2 * nid + 1
 * We make the assumption that the size of the original graph is less than
 * maxof(nid_t) / 2.
 * Also assumes that node ids begin at 1 and increases sequentially.
 */

class BidirectedGraphGPU {
private: // Hide helper functions
    void copy_to_GPU(HandleGraph& host_graph);

#ifndef DEBUG_HOST
    /** Host adjacency and neighbor_start to copy to GPU mem */
    nid_t *h_adjacency;
    nid_t *h_neighbor_start;
#endif /* DEBUG_HOST */

public: // Otherwise this is a glorified struct
    nid_t *adjacency;      /// List of neighbors for all node sides
    nid_t *neighbor_start; /// Where the neighbor list begins for a node side
    size_t size;           /// NODE count of the graph

#ifdef DEBUG_HOST
    nid_t *h_adjacency;
    nid_t *h_neighbor_start;
#endif /* DEBUG_HOST */

    BidirectedGraphGPU(HandleGraph& host_graph);

    void dealloc();
};

#endif /* BIDIRECTEDGRAPHGPU_HPP */
