#include <iostream>
#include <string>
#include <fstream>

#include "../../src/BidirectedGraph.hpp"
#include "../../src/BidirectedGraphGPU.hpp"

using namespace std;

int main(int argc, char* argv[]) {
    string filename = argv[argc - 1];
    ifstream json_file(filename, ifstream::binary);

    BidirectedGraph g;
    bool ret = g.deserialize(json_file);

    if (!ret) {
        std::cerr << "Deserialization error" << std::endl;
        return EXIT_FAILURE;
    }

    BidirectedGraphGPU gpu_g(g);
    std::cout << "Graph" << std::endl;
    size_t gpu_idx;
    for (size_t nid = 0; nid < gpu_g.size; nid++) {
        gpu_idx = 2 * nid;
        std::cout << (nid + 1) << "l" << " " << gpu_idx << std::endl;
        for (size_t c = 0; c < gpu_g.neighbor_count[gpu_idx]; c++) {
            nid_t child_idx = gpu_g.adjacency[gpu_g.neighbor_start[gpu_idx] + c];
            nid_t nid = child_idx / 2 + 1;
            bool is_reverse = child_idx % 2 == 0;
            std::cout << "| " << nid << (is_reverse ? 'l' : 'r') << " " << child_idx << std::endl; 
        }
        gpu_idx = 2 * nid + 1;
        std::cout << (nid + 1) << "r" << " " << gpu_idx << std::endl;
        for (size_t c = 0; c < gpu_g.neighbor_count[gpu_idx]; c++) {
            nid_t child_idx = gpu_g.adjacency[gpu_g.neighbor_start[gpu_idx] + c];
            nid_t nid = child_idx / 2 + 1;
            bool is_reverse = child_idx % 2 == 0;
            std::cout << "| " << nid << (is_reverse ? 'l' : 'r') << " " << child_idx << std::endl; 
        }
    }

    gpu_g.dealloc();

    return EXIT_SUCCESS;    
}