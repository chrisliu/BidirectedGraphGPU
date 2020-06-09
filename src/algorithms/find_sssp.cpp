#include "find_sssp.hpp"
#include "../error.hpp"

#include <limits>

#ifdef DEBUG_SSSP
#include <iostream> // Debug
#endif /* DEBUG_SSSP */

template <class T>
__device__ inline T d_max(T v1, T v2) {
    return (v1 > v2) ? v1 : v2;
}

template <class T>
__device__ __inline__ T add_one(T val) {
    return d_max(val, val + 1);
}

__global__ void initialize_values(distance_t *distances, bool *frontier, nid_t start,
    size_t size, distance_t inf) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;
    nid_t cur_id;

    /// Set distance to inf and frontier to all false
    for (int i = 0; i < (2 * size + total_threads - 1) / total_threads; i++) {
        cur_id = gid + i * total_threads;
        if (cur_id < 2 * size) {
            distances[cur_id] = inf;
            frontier[cur_id] = false;
        }
    }

    /// On thread initializes the starting value
    if (gid == 0) {
        frontier[start] = true;
        distances[start] = 0;
    }
}

__global__ void sssp_iter1(nid_t *adjacency, nid_t *neighbor_start,
    distance_t *cost, distance_t *update, bool *frontier, size_t size) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;
    for (int i = 0; i < (2 * size + total_threads - 1) / total_threads; i++) {
        nid_t cur_id = gid + i * total_threads;
        if (cur_id < 2 * size && frontier[cur_id]) {
            frontier[cur_id] = false;
            distance_t n_cost = add_one(cost[cur_id]);
            for (int n = neighbor_start[cur_id]; n < neighbor_start[cur_id + 1]; n++) {
                atomicMin(&update[adjacency[n]], n_cost);
            }
        }
    }
}

__global__ void sssp_iter2(distance_t *cost, distance_t *update, 
    bool *frontier, int *has_update, size_t size) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_threads = gridDim.x * blockDim.x;
    for (int i = 0; i < (2 * size + total_threads - 1) / total_threads; i++) {
        nid_t cur_id = gid + i * total_threads;
        if (cur_id < 2 * size) {
            if (update[cur_id] < cost[cur_id]) {
                cost[cur_id] = update[cur_id];
                frontier[cur_id] = true;
                atomicOr(has_update, 1);
            }
            update[cur_id] = cost[cur_id];
        }
    }
}

void find_sssp_gpu(BidirectedGraphGPU& g, unsigned long long int** distances) {
    distance_t *d_distances;
    HANDLE_ERROR(cudaMalloc((void**) &d_distances, 2 * g.size * sizeof(distance_t)));
    distance_t *d_update;
    HANDLE_ERROR(cudaMalloc((void**) &d_update, 2 * g.size * sizeof(distance_t)));
    bool *d_frontier;
    HANDLE_ERROR(cudaMalloc((void**) &d_frontier, 2 * g.size * sizeof(bool)));

    int thread_count = 4;
    initialize_values<<<1, thread_count>>>(d_distances, d_frontier, 1, g.size,
        std::numeric_limits<distance_t>::max());

    HANDLE_ERROR(cudaMemcpy(d_update, d_distances, 2 * g.size * sizeof(distance_t), cudaMemcpyDeviceToDevice));    
    int *d_has_update;
    int has_update = 0;
    HANDLE_ERROR(cudaMalloc((void**) &d_has_update, sizeof(int)));
    do {
        sssp_iter1<<<1, thread_count>>>(g.adjacency, g.neighbor_start,
            d_distances, d_update, d_frontier, g.size);
        has_update = 0;
        HANDLE_ERROR(cudaMemcpy(d_has_update, &has_update, sizeof(int), cudaMemcpyHostToDevice));
        sssp_iter2<<<1, thread_count>>>(d_distances, d_update, d_frontier, d_has_update, g.size);
        HANDLE_ERROR(cudaMemcpy(&has_update, d_has_update, sizeof(int), cudaMemcpyDeviceToHost));
    } while (has_update);

    *distances = (distance_t*) malloc(2 * g.size * sizeof(distance_t));
    HANDLE_ERROR(cudaMemcpy(*distances, d_distances, 2 * g.size * sizeof(distance_t), cudaMemcpyDeviceToHost));

#ifdef DEBUG_SSSP
    distance_t *update = (distance_t*) malloc(2 * g.size * sizeof(distance_t));
    bool *frontier = (bool*) malloc(2 * g.size * sizeof(bool));
    HANDLE_ERROR(cudaMemcpy(update, d_update, 2 * g.size * sizeof(distance_t), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(frontier, d_frontier, 2 * g.size * sizeof(bool), cudaMemcpyDeviceToHost));

    std::cout << "Distances" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << (*distances)[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Update" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << update[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Frontier" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << frontier[i] << " ";
    }
    std::cout << std::endl;
    free(update);
    free(frontier);
#endif /* DEBUG_SSSP */
    HANDLE_ERROR(cudaFree(d_distances));
    HANDLE_ERROR(cudaFree(d_update));
    HANDLE_ERROR(cudaFree(d_frontier));
    HANDLE_ERROR(cudaFree(d_has_update));
}