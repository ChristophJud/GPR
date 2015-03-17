/*
 * Copyright 2015 Christoph Jud (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <memory>
#include <utility>
#include <vector>

#include "GaussianProcess.h"
#include "Kernel.h"

typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType         VectorType;
typedef GaussianProcessType::MatrixType         MatrixType;

typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;

typedef std::vector<VectorType>                 TestVectorType;

// parsing data
TestVectorType GetTestData(const std::string& filename){
    TestVectorType test_vectors;

    bool parse = true;

    unsigned input_dimension = 0;

    std::ifstream infile;
    try{
        infile.open(filename);
    }
    catch(...){
        throw std::string("GetTestData: could not read input file");
    }

    std::string line;

    // read header
    if(std::getline(infile, line)){
        std::istringstream iss(line);

        if (!(iss >> input_dimension)) { throw std::string("GetTestData: could not read header"); }
    }
    else{
        throw std::string("GetTestData: could not read header");
    }

    // read rest of the data
    while (std::getline(infile, line))
    {
        VectorType v_input = VectorType::Zero(input_dimension);

        std::istringstream iss(line);
        for(unsigned i=0; i<input_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_input[i] = number;
        }

        test_vectors.push_back(v_input);
    }
    if(!parse) throw std::string("GetTestData: error in parsing data.");

    return test_vectors;

}

void SavePrediction(const TestVectorType& vectors, const std::string& filename){

}

int main (int argc, char *argv[]){
    std::cout << "Gaussian process prediction app:" << std::endl;

    if(argc!=4){
        std::cout << "Usage: " << argv[0] << " gp_prefix input.csv output.csv" << std::endl;
        return -1;
    }

    std::string gp_prefix = argv[1];
    std::string input_filename = argv[2];
    std::string output_filename = argv[3];

    try{
        typedef gpr::WhiteKernel<double>            WhiteKernelType;
        typedef std::shared_ptr<WhiteKernelType>    WhiteKernelTypePointer;
        WhiteKernelTypePointer wk(new WhiteKernelType(1)); // dummy kernel
        GaussianProcessTypePointer gp(new GaussianProcessType(wk));
        gp->Load(gp_prefix);

        TestVectorType test_vectors = GetTestData(input_filename);
        TestVectorType output_vectors;
        for(const auto v : test_vectors){
            output_vectors.push_back(gp->Predict(v));
        }
        SavePrediction(output_vectors, output_filename);
    }
    catch(std::string &s){
        std::cout << "Error: " << s << std::endl;
        return -1;
    }


    return 0;
}

