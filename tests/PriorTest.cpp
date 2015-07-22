
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
#include <memory>
#include <ctime>

#include "Prior.h"

using namespace gpr;


void Test1(){
    /*
     * Test 1: LogGaussian Density
     */
    std::cout << "Test 1: LogGaussian density...\t\t" << std::flush;

    typedef LogGaussianDensity<double> LogGaussianDensityType;

    double mode = 1.5;
    double variance = 0.1;

    LogGaussianDensityType::ParameterPairType mu_sigma =  LogGaussianDensityType::GetMuAndSigma(mode, variance);

    LogGaussianDensityType* p = new LogGaussianDensityType(mu_sigma.first, mu_sigma.second);
    if(std::fabs(p->mode()-mode) > 1e-8){
        std::cout << "[failed] with an error of mode of " << std::fabs(p->mode()-mode) << std::endl;
    }
    else if(std::fabs(p->variance()-variance) > 1e-8){
        std::cout << "[failed] with an error of variance of " << std::fabs(p->variance()-variance) << std::endl;
    }
    else{
        std::cout << "[passed]" << std::endl;
    }
}

void Test2(bool bisection){
    /*
     * Test 2: Inverse Gaussian Density
     */
    if(bisection){
        std::cout << "Test 2.1: Inverse Gaussian density parameter estimation (Bisection method)...\t\t" << std::flush;
    }
    else{
        std::cout << "Test 2.1: Inverse Gaussian density parameter estimation (Halley method)...\t\t" << std::flush;
    }

     typedef InverseGaussianDensity<long double> InverseGaussianDensityType;

     for(double mode=0.05; mode<5; mode+=0.05){
         for(double variance=0.01; variance<2; variance+=0.01){
             InverseGaussianDensityType::ParameterPairType lambda_mu;
             if(bisection){
                 lambda_mu = InverseGaussianDensityType::GetMeanAndLambda(mode, variance, InverseGaussianDensityType::Bisection);
             }
             else{
                 lambda_mu = InverseGaussianDensityType::GetMeanAndLambda(mode, variance, InverseGaussianDensityType::Halley);
             }
             InverseGaussianDensityType* p = new InverseGaussianDensityType(lambda_mu.first, lambda_mu.second);

             if(std::fabs(p->variance()-variance)>1e-10){
                 std::cout << "[failed] with an error of variance of " << std::fabs(p->variance()-variance) << std::endl;
                 return;
             }
             else if(std::fabs(p->mode()-mode)>1e-10){
                 std::cout << "[failed] with an error of mode of " << std::fabs(p->mode()-mode) << std::endl;
                 return;
             }
             delete p;
         }
     }
     std::cout << "[passed]" << std::endl;
}

int main (int argc, char *argv[]){
    std::cout << "Gaussian process regression with different inversion methods: " << std::endl;
    try{
        Test1();
        Test2(false);
        Test2(true);

    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}

