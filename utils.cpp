#include "utils.hpp"
#include <algorithm>
#include <iostream>
#include <fstream>


long double logSum(long double a, long double b){
    long double x = a > b ? a : b;
    long double y = a < b ? a : b;
    return x + logl(1 + expl(y - x));
}


long double logSum(const std::vector<long double>&xs){
    long double max = *std::max_element(xs.begin(), xs.end());
    if (std::isinf(-max)){
        return -INFINITY;
    }

    long double sum = 0;
    for (auto it = xs.begin(); it != xs.end(); ++it){
        sum += expl(*it - max);
    }
    
    return logl(sum) + max;
}


void logSumColumn(const std::vector<std::vector<long double>>&xs,
                  std::vector<long double>&sum){
    std::vector<long double>max(xs[0].size(), -INFINITY);

    for (auto i = 0u; i < xs.size(); ++i){
        for (auto j = 0u; j < xs[0].size(); ++j){
            max[j] = max[j] > xs[i][j] ? max[j] : xs[i][j];
        }
    }
    
    for (auto i = 0u; i < xs.size(); ++i){
        for (auto j = 0u; j < xs[0].size(); ++j){
            sum[j] += expl(xs[i][j] - max[j]);
        }
    }

    for (auto j = 0u; j < xs[0].size(); ++j){
        sum[j] = logl(sum[j]) + max[j];
    }

}



void loadSeq(const std::string&filename, std::vector<std::vector<int>>&seqs){
    std::fstream f;
    f.open(filename, std::ios::in);
    std::string line;
    char tmp;
    f >> tmp;            
    while (!f.eof()){
        seqs.push_back(std::vector<int>());
        while (tmp != '\n' && !f.eof()){            
            seqs.back().push_back(tmp - 'A');
            tmp = f.get();
        }
        tmp = f.get();
    }
}


void dumpSeq(const std::string&filename,
             const std::vector<std::vector<int>>&seqs){
    std::fstream f;
    f.open(filename, std::ios::out);
    for (auto seq = seqs.begin(); seq != seqs.end(); ++seq){
        for (auto l = seq -> begin(); l != seq -> end(); ++l){
            f << *l + 'A';
        }
        f << std::endl;
    }
    f.close();
}



long double compare(const std::vector<std::vector<int>>&a,
                    const std::vector<std::vector<int>>&b){
    long double same = 0;
    long double all = 0;

    if ( a.size() != b.size() ){
        return -1;
    }
    
    for (auto i = 0u; i < a.size(); ++i){
        if (a[i].size() != b[i].size()){
            return -1;
        }
        for (auto j = 0u; j < b.size(); ++j){                        
            same += a[i][j] == b[i][j] ? 1 : 0;
            all += 1;
        }
    }
    return same / all;
}


void dumpAnswer(const std::string&filename, const std::vector<int>&answer){
    std::fstream f;
    f.open(filename, std::ios::out);
    for (auto it = answer.begin(); it != answer.end(); ++it){
        f << "model_0" << answer[*it] << std::endl;
    }    
}
    
void loadAnswer(const std::string&filename, std::vector<int>&answer);
