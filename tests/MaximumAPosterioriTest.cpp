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
#include <cmath>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/FFT>

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"
#include "Likelihood.h"
#include "LikelihoodUtils.h"
#include "Prior.h"
#include "MatrixIO.h"

// typedefinitions
typedef gpr::Kernel<double>                     KernelType;
typedef std::shared_ptr<KernelType>             KernelTypePointer;
typedef gpr::WhiteKernel<double>                 WhiteKernelType;
typedef std::shared_ptr<WhiteKernelType>        WhiteKernelTypePointer;
typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;
typedef gpr::PeriodicKernel<double>             PeriodicKernelType;
typedef std::shared_ptr<PeriodicKernelType>     PeriodicKernelTypePointer;
typedef gpr::SumKernel<double>                  SumKernelType;
typedef std::shared_ptr<SumKernelType>          SumKernelTypePointer;
typedef gpr::GaussianProcess<double>            GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;


void Test1(){
    /*
     * Test 1: perform maximum aposteriori
     */
    std::cout << "Test 1: maximum log gaussian likelihood + log gaussian prior (gradient descent)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    auto f = [](double x)->double { return std::sin(x)*std::cos(2.2*std::sin(x)); };

    double val = 0;
    unsigned n = 30;
    MatrixType signal = MatrixType::Zero(2,n);
    for(unsigned i=0; i<n; i++){
        signal(0,i) = val;
        signal(1,i) = f(val);
        val += 15.0/n;
    }

    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0.01); // zero since with the white kernel (see later) this is considered

    for(unsigned i=0; i<n; i+=10){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }


    double p_sigma = 1;
    double p_scale = 0.4;
    //double p_period = M_PI/2;
    double p_period = M_PI/1.8;
    double g_sigma = 100;
    double g_scale = 0.2;

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

    // construct Gaussian priors
    typedef gpr::GaussianDensity<double> GaussianDensityType;
    typedef std::shared_ptr<GaussianDensityType> GaussianDensityTypePointer;

    std::vector<GaussianDensityTypePointer> g_densities;
    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_scale, 0.2)));
    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_period, 0.05)));
    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_sigma, 2)));
    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(g_sigma, 10)));
    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(g_scale, 0.05)));


    double lambda = 1e-3;
    for(unsigned i=0; i<400; i++){
        // analytical
        try{
            PeriodicKernelTypePointer pk(new PeriodicKernelType(p_scale, p_period, p_sigma));
            GaussianKernelTypePointer gk(new GaussianKernelType(g_sigma, g_scale));
            SumKernelTypePointer sk(new SumKernelType(pk,gk));
            gp->SetKernel(sk);

            VectorType likelihood_update = gl->GetParameterDerivatives(gp);

            std::cout << (*gl)(gp) << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
            //std::cout << "log gauss: scale/period/sigma, sigma/scale: " << (*g_densities[0])(p_scale) << "/" << (*g_densities[1])(p_period) << "/" << (*g_densities[2])(p_sigma);
            //std::cout << ", " << (*g_densities[3])(g_sigma) << "/" << (*g_densities[4])(g_scale) << std::endl;


            p_scale += lambda * (likelihood_update[0] + g_densities[0]->GetLogDerivative(p_scale));
            p_period += lambda * (likelihood_update[1] + g_densities[1]->GetLogDerivative(p_period));
            p_sigma += lambda * (likelihood_update[2] + g_densities[2]->GetLogDerivative(p_sigma));
            g_sigma += lambda * (likelihood_update[3] + g_densities[3]->GetLogDerivative(g_sigma));
            g_scale += lambda * (likelihood_update[4] + g_densities[4]->GetLogDerivative(g_scale));
        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::endl;
            return;
        }
    }

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=0; i<n; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(signal(0,i));

        std::cout << prediction_y.back() << std::endl;
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 5){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed] with an error of " << err/prediction_x.size() << std::endl;
    }
}


int main (int argc, char *argv[]){
    std::cout << "Gaussian likelihood kernel test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}

