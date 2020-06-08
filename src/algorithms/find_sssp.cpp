#include "find_sssp.hpp"

#include <limits>

template <class T>
__device__ __inline__ add_one(T val) {
    return std::max(val, val + 1);
}

__device__ __inline__ get_complement_id(nid_t nid) {
    return nid + (nid + 1) % 2 - nid % 2;
}

__global__ void initialize_values(nid_t* distances, bool* frontier, nid_t start,
    size_t size) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;
    nid_t cur_id;
    nid_t max = std::numeric_limits<nid_t>::max();

    for (int i = 0; i < (size + total_threads - 1) / total_threads; i++) {
        cur_id = gid + i * total_threads;
        if (cur_id < size * 2)  {
            distances[cur_id] = max;
            frontier[cur_id] = false;
        }
    }

    if (gid == 0) {
        frontier[start] = true;
    }
}

// __global__ void sssp_iterate(nid_t* adjacency, nid_t* neighbor_start,
//     size_t size, nid_t* distances, nid_t* iter_distances, bool* frontier) 
//     nid_t gid = blockIdx.x * threadDim.x + threadIdx.x;
//     size_t total_threads = blockDim.x * threadDim.x;
//     nid_t cur_id;
//     nid_t comp_id;
//     nid_t min_distance;
//     for (int i = 0; i < (size + total_threads - 1)  / total_threads; i++) {
//         cur_id = gid + i * total_threads;
//         if (cur_id < size && frontier[cur_id]) {
//             frontier[cur_id] = false;
//             min_distance = std::numeric_limits<nid_t>::max();

//             comp_id = get_complement_id(cur_id);
//             for (size_t idx = neigbor_start[comp_id]; idx < neigbor_start[comp_id + 1]; idx++) {
//                 min_distance = std::min(min_distance, 
//                     add_one(distances[get_complement_id[adjacency[idx]]]));
//             }
//         }
//     }
// }

// __global__ void sssp_update() {

// }

#include <iostream> // Debug
void find_sssp_gpu(BidirectedGraphGPU& g, nid_t* distances, nid_t nid, bool is_reverse) {
    nid_t* d_distances;
    cudaMalloc((void**) &d_distances, 2 * g.size * sizeof(nid_t));
    bool* d_frontier;
    cudaMalloc((void**) &d_frontier, 2 * g.size * sizeof(bool));

    int thread_count = 2;
    initialize_values<<<1, thread_count>>>(d_distances, d_frontier, 1, g.size);

    cudaMemcpy(distances, d_distances, 2 * g.size * sizeof(nid_t), cudaMemcpyDeviceToHost);
    bool* frontier;
    cudaMemcpy(frontier, d_frontier, 2 * g.size * sizeof(bool), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << distances[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << frontier[i] << " ";
    }
    std::cout << std::endl;
}