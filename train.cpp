#include <iostream>
#include <algorithm>
#include "utils.hpp"
#include "hmm.hpp"

int main(int argc, char** argv){
    if (argc < 4){
        std::cerr << "Insufficient number of argument." << std::endl;
        return -1;
    }
    unsigned nIter = stoi(std::string(argv[1]));

    HMM hmm = HMM();
    hmm.load(std::string(argv[2]));

    std::vector<std::vector<int>>observeds;
    loadSeq(std::string(argv[3]), observeds);
    std::cout << "Finish loading" << std::endl;

    hmm.train(nIter, observeds);

    hmm.dump(std::string(argv[4]));
}
