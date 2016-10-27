#include "hmm.hpp"
#include "utils.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>


void print2D(const std::vector<std::vector<long double>>&x){
    for (int i = 0; i < x.size(); ++i){
        for (int j = 0; j < x[i].size(); ++j ){
            std::cout << x[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

void print2D(const std::vector<long double>&x){
    for (int i = 0; i < x.size(); ++i){
            std::cout << x[i] << " ";
    }
        std::cout << std::endl;
}


HMM::HMM(){};

void HMM::load(const std::string&initFile){
    std::fstream f;
    f.open(initFile, std::ios::in);
    
    while( !f.eof() ){
        std::string buffer;
        f >> buffer;
        if (buffer == "" || buffer == "") continue;

        if (buffer == "initial:"){
            f >> nStates;
            startprob = std::vector<long double>(nStates);
            for (auto i = 0u ; i < nStates ; i++) {
                f >> startprob[i];
                startprob[i] = logl(startprob[i]);
            }
        }
        else if( buffer == "transition:"){
            f >> nStates;
            transprob = std::vector<std::vector<long double>>
                (nStates, std::vector<long double>(nStates));
            
            for (auto i = 0u; i < nStates; i++){
                for (auto j = 0u; j < nStates; j++){
                    f >> transprob[i][j];
                    transprob[i][j] = logl(transprob[i][j]);
                }
            }
        }
        else if (buffer == "observation:"){
            f >> nStates;
            outputprob = std::vector<std::vector<long double>>
                (nStates, std::vector<long double>(nStates));

            for (auto i = 0u; i < nStates; i++){
                for (auto j = 0u; j < nStates; j++){
                    f >> outputprob[j][i];
                    outputprob[j][i] = logl(outputprob[j][i]);
                }
            }
        }
    }
}


void HMM::dump(const std::string&outFile){
    std::fstream f;
    f.open(outFile, std::ios::out);
    
    f << "initial: " << nStates << std::endl;
    for (auto i = 0u ; i < nStates - 1; ++i){
        f << std::setprecision(5) << expl(startprob[i]) << " ";
    }
    f << std::setprecision(5) << expl(startprob[nStates - 1]) << std::endl;
    
    f << "\ntransition: " << nStates << std::endl;
    for (auto i = 0u; i < nStates; i++){
        for( auto j = 0u; j < nStates - 1; j++ ){
            f << std::setprecision(5) << expl(transprob[i][j]) << " "; 
        }
        f << std::setprecision(5) << expl(transprob[i][nStates - 1]) << std::endl;
    }

    f << "\nobservation: " << nStates << std::endl;
    for (auto i = 0u; i < nStates; i++){
        for (auto j = 0u; j < nStates - 1; j++){
            f << std::setprecision(5) << expl(outputprob[j][i]) << " ";
        }
        f << std::setprecision(5) << expl(outputprob[nStates - 1][i]) << std::endl;
   }

}

void HMM::forward(const std::vector<int>&observed){

    for (auto s = 0u; s < nStates; ++s ){
        alpha[0][s] = startprob[s] + outputprob[s][observed[0]];
    }

    for (auto i = 1u; i < observed.size(); ++i){
        for (auto s2 = 0u; s2 < nStates; ++s2){
            std::vector<long double>probsFrom(nStates);
            for (auto s1 = 0u; s1 < nStates; ++s1){
                probsFrom[s1] = alpha[i - 1][s1] + transprob[s1][s2] + outputprob[s2][observed[i]];
            }
            alpha[i][s2] = logSum(probsFrom);
        }
    }
}

void HMM::backward(const std::vector<int>&observed){

    int nStates = transprob.size();
    for (int s = 0; s < nStates; ++s ){
        beta[observed.size() - 1][s] = 0;
    }

    for (int i = observed.size() - 2; i >= 0; --i){
        for (int s1 = 0; s1 < nStates; ++s1){
            std::vector<long double>probsFrom(nStates);
            for (int s2 = 0; s2 < nStates; ++s2){
                probsFrom[s2] = beta[i + 1][s2] + transprob[s1][s2] + outputprob[s2][observed[i + 1]];
            }
            beta[i][s1] = logSum(probsFrom);
        }
    }
}


HMM::Accumulator::Accumulator(const unsigned int nStates) {
    this -> nStates = nStates;
    s = std::vector<long double>(nStates, -INFINITY);
    gammaSum = std::vector<long double>(nStates, -INFINITY);
    t = std::vector<std::vector<long double>>(nStates, std::vector<long double>(nStates, -INFINITY)); 
    o = std::vector<std::vector<long double>>(nStates, std::vector<long double>(nStates, -INFINITY)); 
    xiSum = std::vector<std::vector<long double>>(nStates, std::vector<long double>(nStates, -INFINITY)); 
}

void HMM::Accumulator::computeGamma(const std::vector<std::vector<long double>>&alpha,
                                    const std::vector<std::vector<long double>>&beta,
                                    std::vector<std::vector<long double>>&gamma){
    int nStates = alpha[0].size();
    for (unsigned int i = 0; i < alpha.size(); ++i){
        for (int s = 0; s < nStates; ++s){
            gamma[i][s] = alpha[i][s] + beta[i][s];
        }
        long double sum = logSum(gamma[i]);
        
        for (int s = 0; s < nStates; ++s){
            gamma[i][s] = gamma[i][s] - sum;
        }
    }
}


void HMM::Accumulator::computeXi(const std::vector<long double>&alphat,
                                 const std::vector<long double>&betat1,
                                 const std::vector<long double>&ot1prob,
                                 const std::vector<std::vector<long double>>&transprob,
                                 std::vector<std::vector<long double>>&xi){
    int nSymbols = alphat.size();
    std::vector<long double>rowSum(nSymbols);
    for (int i = 0; i < nSymbols; ++i ){
        for (int j = 0; j < nSymbols; ++j ){
            xi[i][j] = alphat[i] + transprob[i][j] + ot1prob[j] + betat1[j];
        }
        rowSum[i] = logSum(xi[i]);
    }
    long double sum = logSum(rowSum);

    for (int i = 0; i < nSymbols; ++i ){
        for (int j = 0; j < nSymbols; ++j ){
            xi[i][j] = xi[i][j] - sum;
        }
    }
}


void HMM::Accumulator::accumulate(const std::vector<int>&observed,
                                  const std::vector<std::vector<long double>>&alpha,
                                  const std::vector<std::vector<long double>>&beta,
                                  const std::vector<std::vector<long double>>&transprob,
                                  const std::vector<std::vector<long double>>&outputprob){
    std::vector<std::vector<long double>>gamma(observed.size(), std::vector<long double>(nStates, 0));
    computeGamma(alpha, beta, gamma);
    // std::cout << "gamma" << std::endl;
    // print2D(gamma);
    std::vector<long double>tmp(nStates, 0);
    logSumColumn(gamma, tmp);

    for (unsigned int i = 0; i < nStates; ++i){
        gammaSum[i] = logSum(gammaSum[i], tmp[i]);
    }
    
    std::cout << "gammaSum" << std::endl;
    print2D(gammaSum);
    
    // Start Probability
    for (auto i = 0u; i < nStates; ++i){
        s[i] = logSum(s[i], gamma[0][i]);
    }
    
    // Transition probability
    for (auto t = 0u; t < observed.size() - 1; ++t ){
        std::vector<long double>ot1prob(nStates);
        for (auto s = 0u; s < nStates; ++s ){
            ot1prob[s] = outputprob[s][observed[t + 1]];
        }
        std::vector<std::vector<long double>>xit(nStates, std::vector<long double>(nStates));        
        computeXi(alpha[t], beta[t + 1], ot1prob, transprob, xit);

        // std::cout << "xit:" << std::endl;
        // print2D(xiSum);
        // std::cout << std::endl;
        
        for (auto i = 0u; i < nStates; ++i ){
            for (auto j = 0u; j < nStates; ++j ){
                xiSum[i][j] = logSum(xiSum[i][j], xit[i][j]);
            }
        }
    }
    std::cout << "xi:" << std::endl;
    // print2D(xiSum);
    for (int i = 0; i < nStates; ++i){
        std::cout << logSum(xiSum[i]) << ' ';
    }
    std::cout << std::endl;

    
    // std::cout << "xiSum:" << std::endl;
    // print2D(xiSum);
    // std::cout << std::endl;

    // Output probability
    for (auto t = 0u; t < observed.size(); ++t){
        for (auto s = 0u; s < nStates; ++s){
            o[s][observed[t]] =
                logSum(o[s][observed[t]], gamma[t][s]);
        }
    }
}


void HMM::updateProb(const HMM::Accumulator&accumulator){
    // Start probability
    auto startSum = logSum(accumulator.s);
    for (auto i = 0u; i < nStates; ++i){
        startprob[i] = accumulator.s[i] - startSum;
    }
    
    // Transition probability
    for (auto i = 0u; i < nStates; ++i){
        for (auto j = 0u; j < nStates; ++j){
            transprob[i][j] = accumulator.xiSum[i][j] - accumulator.gammaSum[i];
        }
    }

    // std::cout << "trans:" << std::endl;
    // print2D(transprob);
        
    // Output probability
    for (auto i = 0u; i < nStates; ++i){
        for (auto j = 0u; j < nStates; ++j){
            outputprob[i][j] = accumulator.o[i][j] - accumulator.gammaSum[i];
        }
    }
}
    

void HMM::train(const unsigned int nIter,
                const std::vector<std::vector<int>>&observed){
    for (auto i = 0u; i < nIter; ++i){
        std::cout << "E-Step iter: " << i << std::endl;
        HMM::Accumulator accumulator(nStates);
        alpha = std::vector<std::vector<long double>>(observed[0].size(), std::vector<long double>(nStates));
        beta = std::vector<std::vector<long double>>(observed[0].size(), std::vector<long double>(nStates));
        for (auto j = 0u; j < observed.size(); ++j){
        // for (auto j = 0u; j < 5; ++j){
            // E-step
            // std::cout << "forward " << j << std::endl;
            forward(observed[j]);
            backward(observed[j]);
            accumulator.accumulate(observed[j], alpha, beta, transprob, outputprob);
        }
        std::cout << "M-Step iter: " << i << std::endl;        
        updateProb(accumulator);
    }
}

