#include "group_irreducible.hpp"
#include "../error.hpp"

#include <limits>

#include <iostream>

template <class T>
__device__ inline T d_min(T v1, T v2) {
    return (v1 < v2) ? v1 : v2;
}

__device__ inline nid_t complement_nid(nid_t nid) {
    return nid + (nid + 1) % 2 - nid % 2;
}

__global__ void initialize_membership(nid_t *membership, nid_t *reducible, bool *reducible_mask,
    size_t count, nid_t max, const size_t size) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_count = gridDim.x * blockDim.x;
    for (int i = 0; i < (2 * size + thread_count - 1) / thread_count; i++) {
        nid_t cur_id = gid + i * thread_count;
        if (cur_id < 2 * size) {
            membership[cur_id] = max;
            reducible_mask[cur_id] = false;
        }
    }

    if (gid < count) {
        membership[reducible[gid]] = reducible[gid];
        reducible_mask[reducible[gid]] = true;
    }
}

__global__ void group_irreducible1(nid_t *adjacency, nid_t *neighbor_start, nid_t *membership, nid_t *membership_update, size_t size) {
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_count = gridDim.x * blockDim.x;
    for (int i = 0; i < (2 * size + thread_count - 1) / thread_count; i++) {
        nid_t cur_id = gid + i * thread_count;
        if (cur_id < 2 * size) {
            nid_t id_mem = membership[cur_id];
            for (int n = neighbor_start[cur_id]; n < neighbor_start[cur_id + 1]; n++) {
                id_mem = d_min(id_mem, membership[complement_nid(adjacency[n])]);
            }
            membership_update[cur_id] = id_mem;
        }
    }
}

__global__ void group_irreducible2(nid_t *membership, nid_t *membership_update, bool *reducible_mask, size_t size, int *has_update) {
    bool updated = false;
    nid_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t thread_count = gridDim.x * blockDim.x;
    for (int i = 0; i < (size + thread_count - 1) / thread_count; i++) {
        nid_t cur_id = gid + i * thread_count;
        if (cur_id < size) {
            cur_id *= 2;
            if (!reducible_mask[cur_id]) { // Irreducible
                nid_t min = d_min(membership_update[cur_id], membership_update[cur_id + 1]);
                membership[cur_id] = min;
                membership[cur_id + 1] = min;
                updated |= min != membership_update[cur_id] || min != membership_update[cur_id + 1];
            } else { // Reducible
                updated |= membership[cur_id] != membership_update[cur_id] || membership[cur_id + 1] != membership_update[cur_id + 1];
                membership[cur_id] = membership_update[cur_id];
                membership[cur_id + 1] = membership_update[cur_id + 1];
            }
        }
    }

    if (updated) {
        atomicOr(has_update, 1);
    }
}

void group_irreducible(BidirectedGraphGPU& g, nid_t *reducible, size_t count, nid_t **membership) {
    nid_t *d_reducible;
    HANDLE_ERROR(cudaMalloc((void**) &d_reducible, count * sizeof(nid_t)));
    HANDLE_ERROR(cudaMemcpy(d_reducible, reducible, count * sizeof(nid_t), cudaMemcpyHostToDevice));

    bool *d_reducible_mask;
    nid_t *d_membership;
    nid_t *d_membership_update;
    HANDLE_ERROR(cudaMalloc((void**) &d_reducible_mask, 2 * g.size * sizeof(bool)));
    HANDLE_ERROR(cudaMalloc((void**) &d_membership, 2 * g.size * sizeof(nid_t)));
    HANDLE_ERROR(cudaMalloc((void**) &d_membership_update, 2 * g.size * sizeof(nid_t)));

    initialize_membership<<<1, 10>>>(d_membership, d_reducible, d_reducible_mask,
        count, std::numeric_limits<nid_t>::max(), g.size);
    int *d_has_update;
    int has_update = 0;
    HANDLE_ERROR(cudaMalloc((void**) &d_has_update, sizeof(int)));

    do {
        group_irreducible1<<<1, 10>>>(g.adjacency, g.neighbor_start, d_membership, d_membership_update, g.size);
        has_update = 0;
        HANDLE_ERROR(cudaMemcpy(d_has_update, &has_update, sizeof(int), cudaMemcpyHostToDevice));
        group_irreducible2<<<1, 10>>>(d_membership, d_membership_update, d_reducible_mask, g.size, d_has_update);    
        HANDLE_ERROR(cudaMemcpy(&has_update, d_has_update, sizeof(int), cudaMemcpyDeviceToHost));
    } while (has_update);

    *membership = (nid_t*) malloc(2 * g.size * sizeof(nid_t));
    HANDLE_ERROR(cudaMemcpy(*membership, d_membership, 2 * g.size * sizeof(nid_t), cudaMemcpyDeviceToHost));

#ifdef DEBUG_GROUP_IRREDUCIBLE
    bool *reducible_mask = (bool*) malloc(2 * g.size * sizeof(bool));
    nid_t *membership_update = (nid_t*) malloc(2 * g.size * sizeof(nid_t));
    HANDLE_ERROR(cudaMemcpy(reducible_mask, d_reducible_mask, 2 * g.size * sizeof(bool), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaMemcpy(membership_update, d_membership_update, 2 * g.size * sizeof(nid_t), cudaMemcpyDeviceToHost));


    std::cout << "Updated: " << has_update << std::endl;

    std::cout << "Reducible Mask" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << reducible_mask[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Membership" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << (*membership)[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "Membership Update" << std::endl;
    for (int i = 0; i < 2 * g.size; i++) {
        std::cout << membership_update[i] << " ";
    }
    std::cout << std::endl;
    free(reducible_mask);
    free(membership_update);
#endif /* DEBUG_GROUP_IRREDUCIBLE */

    HANDLE_ERROR(cudaFree(d_reducible));    
    HANDLE_ERROR(cudaFree(d_reducible_mask));    
    HANDLE_ERROR(cudaFree(d_membership));    
    HANDLE_ERROR(cudaFree(d_membership_update));    
    HANDLE_ERROR(cudaFree(d_has_update));
}