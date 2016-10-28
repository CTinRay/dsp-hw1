#include <iostream>
#include <vector>
#include <fstream>
#include "hmm.hpp"
#include "utils.hpp"


int main(int argc, char**argv){

    if (argc < 4) {
        std::cout << "insufficient number of argument." << std::endl;
        return -1;
    }
    
    // Load models
    std::vector<HMM>hmms;
    std::vector<std::string>modelNames;
    std::fstream f;
    f.open(std::string(argv[1]), std::ios::in);
    std::string line;
    std::getline(f, line);
    while (!f.eof()){
        hmms.push_back(HMM());
        hmms.back().load(line);
        modelNames.push_back(line);
        std::getline(f, line);
    }
    f.close();

    std::cout << "finish loading data" << std::endl;
    
    // Load test data
    std::vector<std::vector<int>>seqs;
    loadSeq(std::string(argv[2]), seqs);

    
    std::vector<std::string>results(seqs.size());
    for (auto i = 0u; i < seqs.size(); ++i){
        long double maxProb = -INFINITY;
        int model = -1;
        for (auto m = 0u; m < hmms.size(); ++m){
            auto prob = hmms[m].test(seqs[i]);
            if (prob > maxProb) {
                maxProb = prob;
                model = m;
            }
        }
        results[i] = modelNames[model];
    }
    
    f.open(std::string(argv[3]), std::ios::out);
    for (auto it = results.begin(); it != results.end(); ++it){
        f << *it << std::endl;
    }
    
    f.close();
            
    if (argc > 4){
        std::vector<std::string>answer;
        f.open(std::string(argv[4]), std::ios::in);
        int same = 0;
        std::string line;
        for (auto it = results.begin(); it != results.end(); ++it){
            std::getline(f, line);
            same += *it == line ? 1 : 0;
        }
        std::cout << "accuracy:" << (double) same / (double) results.size() << std::endl;
    }
}
