#include <iostream>
#include <string>
#include <fstream>
#include <vector>

#include "../../src/BidirectedGraph.hpp"
#include "../../src/BidirectedGraphGPU.hpp"
#include "../../src/algorithms/group_irreducible.hpp"

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

    nid_t *membership;
    // Nids 1, 5, 6, 7, 9 for reducible1.json
    std::vector<nid_t> reducible = {0, 1, 8, 9, 10, 11, 12, 13, 16, 17};
    group_irreducible(gpu_g, reducible.data(), (size_t) 10, &membership);

    gpu_g.dealloc();

    return EXIT_SUCCESS;    
}