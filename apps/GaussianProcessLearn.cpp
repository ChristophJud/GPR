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
#include "KernelFactory.h"
#include "Kernel.h"

typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;
typedef GaussianProcessType::VectorType         VectorType;
typedef GaussianProcessType::MatrixType         MatrixType;

typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;

typedef gpr::KernelFactory<double>              KernelFactoryType;

typedef std::pair<VectorType, VectorType>       TrainingPairType;
typedef std::vector<TrainingPairType>           TrainingPairVectorType;

// parsing data
TrainingPairVectorType GetTrainingData(const std::string& filename){
    TrainingPairVectorType train_pairs;

    bool parse = true;

    unsigned input_dimension = 0;
    unsigned output_dimension = 0;

    std::ifstream infile;
    try{
        infile.open(filename);
    }
    catch(...){
        throw std::string("GetTrainingData: could not read input file");
    }

    std::string line;

    // read header
    if(std::getline(infile, line)){
        std::istringstream iss(line);

        if (!(iss >> input_dimension >> output_dimension)) { throw std::string("GetTrainingData: could not read header"); } // error
    }
    else{
        throw std::string("GetTrainingData: could not read header");
    }

    // read rest of the data
    while (std::getline(infile, line))
    {
        VectorType v_input = VectorType::Zero(input_dimension);
        VectorType v_output = VectorType::Zero(output_dimension);

        std::istringstream iss(line);
        for(unsigned i=0; i<input_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_input[i] = number;
        }

        for(unsigned i=0; i<output_dimension; i++){
            double number;
            if (!(iss >> number)) { parse=false; break; }
            v_output[i] = number;
        }

        train_pairs.push_back(std::make_pair(v_input, v_output));
    }
    if(!parse) throw std::string("GetTrainingData: error in parsing data.");

    return train_pairs;

}

int main (int argc, char *argv[]){
    std::cout << "Gaussian process training app:" << std::endl;

    if(argc!=5){
        std::cout << "Usage: " << argv[0] << " data.csv kernel_string data_noise output_gp" << std::endl;

        std::cout << std::endl << "Example of a kernel string: GaussianKernel(2.3, 1.0,)" << std::endl;
        std::cout << "Example of an input file:" << std::endl;
        std::cout << "4 2" << std::endl;
        std::cout << "x0 x1 x2 x3 y0 y1" << std::endl;
        std::cout << " .... " << std::endl;
        return -1;
    }

    std::string data_filename = argv[1];
    std::string kernel_string = argv[2];
    double gp_sigma;
    std::stringstream ss; ss << argv[3]; ss >> gp_sigma;
    std::string output_prefix = argv[4];


    std::cout << "Configuration: " << std::endl;
    std::cout << " - data: " << data_filename << std::endl;
    std::cout << " - kernel string: " << kernel_string << std::endl;
    std::cout << " - data noise: " << gp_sigma << std::endl;
    std::cout << " - output: " << output_prefix << std::endl << std::endl;

    try{
        std::cout << "Parsing data... " << std::flush;
        KernelTypePointer kernel = KernelFactoryType::GetKernel(kernel_string);
        GaussianProcessTypePointer gp(new GaussianProcessType(kernel));
        gp->SetSigma(gp_sigma);

        std::cout << "[done]" << std::endl << "Build Gaussian process... " << std::flush;
        TrainingPairVectorType train_pairs = GetTrainingData(data_filename);
        for(const auto &tp : train_pairs){
            gp->AddSample(tp.first, tp.second);
        }

        std::cout << "[done]" << std::endl << "Perform learning... " << std::flush;
        gp->Initialize();
        std::cout << "[done]" << std::endl << "Saving Gaussian process... " << std::flush;
        gp->Save(output_prefix);
        std::cout << "[done]" << std::endl;
    }
    catch(std::string& s){
        std::cout << std::endl << "Error: " << s << std::endl;
        return -1;
    }

    return 0;
}
