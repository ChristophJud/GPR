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

// constructs a kernel based on randomly sampled parameters
// the kernel is:
// Gaussian + Periodic + White
// (each parameter is Gamma distributed

// parameters[0] = wscale();
// parameters[1] = pperiod();
// parameters[2] = pscale();
// parameters[3] = psigma();
// parameters[4] = gscale();
// parameters[5] = gsigma();
KernelTypePointer GetKernel(const GaussianProcessType::VectorType& parameters){
    GaussianKernelTypePointer gk(new GaussianKernelType(parameters[5], parameters[4]));
    PeriodicKernelTypePointer pk(new PeriodicKernelType(parameters[2], parameters[1], parameters[3]));
    SumKernelTypePointer sk(new SumKernelType(gk,pk));
    WhiteKernelTypePointer wk(new WhiteKernelType(parameters[0]));
    SumKernelTypePointer kernel(new SumKernelType(sk,wk));
    return kernel;
}

void Test1(){
    /*
     * Test 1.1: perform maximum likelihood
     *      - Sum kernel of a Gaussian and a Period kernel is used
     */
    std::cout << "Test 1: maximum log gaussian likelihood... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    MatrixType signal;
    try{
        signal = gpr::ReadMatrix<MatrixType>("/home/jud/Projects/GaussianProcessRegression/install/data/breathing1D.mat");
    }
    catch(...){
        std::cout << "[failed] could not read ../data/breathing1D.mat" << std::endl;
        return;
    }


    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0); // zero since with the white kernel (see later) this is considered

    for(unsigned i=80; i<700; i++){
        VectorType x = VectorType::Zero(1); x[0] = i;
        VectorType y = VectorType::Zero(1); y[0] = signal(0,i);
        gp->AddSample(x,y);
    }
    //gp->DebugOn();

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());


    // mean estimate of periodicity parameter
    double p_estimate =  gpr::GetLocalPeriodLength<double>(signal.transpose(), 10);


    /*
     * construct/define prior over the parameters
     *   - GammaDensity to sample parameters
     *   -LogGammaDensity to evaluate prior for the posterior
     */
    typedef gpr::GammaDensity<double>          GammaDensityType;
    typedef gpr::LogGammaDensity<double>       LogGammaDensityType;

    // for period kernel
    LogGammaDensityType lgd_pperiod(p_estimate/0.5, 0.5);
    LogGammaDensityType lgd_pscale(4.5,2);
    LogGammaDensityType lgd_psigma(2.5, 0.5);
    // for gaussian kernel
    LogGammaDensityType lgd_gscale(5,1);
    LogGammaDensityType lgd_gsigma(3,3.5);
    // for white kernel
    LogGammaDensityType lgd_wscale(3,0.4);

    // for period kernel
    GammaDensityType gd_pperiod(p_estimate/0.5, 0.5);
    GammaDensityType gd_pscale(4.5,2);
    GammaDensityType gd_psigma(2.5, 0.5);
    // for gaussian kernel
    GammaDensityType gd_gscale(5,1);
    GammaDensityType gd_gsigma(3,3.5);
    // for white kernel
    GammaDensityType gd_wscale(3,0.4);




    // initial kernel
    VectorType parameters = VectorType::Zero(6);
    parameters[0] = gd_wscale();
    parameters[1] = gd_pperiod();
    parameters[2] = gd_pscale();
    parameters[3] = gd_psigma();
    parameters[4] = gd_gscale();
    parameters[5] = gd_gsigma();

    gp->SetKernel(GetKernel(parameters));

    // begin optimization
    double lambda = 0.011;

    for(unsigned i=0; i<100; i++){

        try{

            VectorType value = (*gl)(gp);
            std::cout << "Iteration: " << i << ", likelihood: " << value;

            VectorType update = gl->GetParameterDerivatives(gp);
            //std::cout << ", gradient: " << update.transpose() << std::endl;



            if(update.rows()!=parameters.rows()){
                std::cout << "[failed] Wrong number of parameters" << std::endl;
            }

            for(unsigned p=0; p<parameters.rows(); p++){
                parameters[p] -= lambda * update[p];
            }

            std::cout << ", parameters: " << parameters.transpose() << std::endl;

            gp->SetKernel(GetKernel(parameters));
        }
        catch(std::string& s){
            std::cout << "Error: " << s << std::endl;
            return;
        }
    }

    std::cout << "Parameters are: " << parameters.transpose() << std::endl;

    return;

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=80; i<3000; i++){
        VectorType x = VectorType::Zero(1); x[0] = i;
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(i);
    }

    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;
    std::cout << "p=np.array([";
    for(unsigned i=0; i<prediction_y.size(); i++){
        std::cout << prediction_y[i] << ", " << std::flush;
    }
    std::cout << "])" << std::endl;
    std::cout << "x=np.array([";
    for(unsigned i=0; i<prediction_x.size(); i++){
        std::cout << prediction_x[i] << ", " << std::flush;
    }
    std::cout << "])" << std::endl;
    std::cout << "gty=np.array([";
    for(unsigned i=0; i<signal.cols(); i++){
        std::cout << signal(0,i) << ", " << std::flush;
    }
    std::cout << "])" << std::endl;
    std::cout << "gtx=np.array([";
    for(unsigned i=0; i<signal.cols(); i++){
        std::cout << i << ", " << std::flush;
    }
    std::cout << "])" << std::endl;
    std::cout << "hx = plt.plot(gtx,gty,'-b', label='ground truth')" << std::endl;
    std::cout << "hx = plt.plot(x,p,'-r', label='prediction')" << std::endl;
    std::cout << "plt.legend()" << std::endl;
    std::cout << "plt.show()" << std::endl;
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
