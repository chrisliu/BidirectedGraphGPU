#include <iostream>
#include <string>
#include <fstream>

#include "../../src/BidirectedGraph.hpp"
#include "../../src/BidirectedGraphGPU.hpp"
#include "../../src/algorithms/find_sssp.hpp"

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

    nid_t* distances = (nid_t*) malloc(2 * gpu_g.size * sizeof(nid_t));
    find_sssp_gpu(gpu_g, distances);
    free(distances);

    gpu_g.dealloc();

    return EXIT_SUCCESS;    
}