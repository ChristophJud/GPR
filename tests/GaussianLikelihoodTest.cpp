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
    std::vector< std::pair< std::string, std::pair<double,double> > > pdf_params;
    pdf_params.push_back(std::make_pair("WScale", std::make_pair(4, 1))); // white kernel
    pdf_params.push_back(std::make_pair("PPeriod", std::make_pair(p_estimate/0.5, 0.5))); // period kernel
    pdf_params.push_back(std::make_pair("PScale", std::make_pair(5 ,2)));
    pdf_params.push_back(std::make_pair("PSigma", std::make_pair(3, 1.5)));
    pdf_params.push_back(std::make_pair("GScale", std::make_pair(10, 0.3))); // gaussian kernel
    pdf_params.push_back(std::make_pair("GSigma", std::make_pair(10, 0.3)));


    typedef gpr::GammaDensity<double>          GammaDensityType;
    typedef gpr::LogGammaDensity<double>       LogGammaDensityType;
    typedef gpr::GaussianDensity<double>        GaussianDensityType;
    typedef gpr::LogGaussianDensity<double>     LogGaussianDensityType;

    std::vector< std::pair<std::string, GammaDensityType*> > pdfs;
    for(auto const &p : pdf_params){
        pdfs.push_back(std::make_pair(p.first, new GammaDensityType(p.second.first, p.second.second)));
    }

    std::vector< std::pair<std::string, LogGammaDensityType*> > logpdfs;
    for(auto const &p : pdf_params){
        logpdfs.push_back(std::make_pair(p.first, new LogGammaDensityType(p.second.first, p.second.second)));
    }

        // initial kernel
    VectorType parameters = VectorType::Zero(pdf_params.size());
    for(unsigned i=0; i<parameters.rows(); i++){
        parameters[i] = (*pdfs[i].second)();
    }

    gp->SetKernel(GetKernel(parameters));

    // begin optimization
    double lambda = 0.011;
    double w = 0.5; // trade off likelihood and prior: 1 -> only likelihood, 0 -> only prior
    if(w>1) w=1;
    if(w<0) w=0;


    double likelihood_value, prior_value;
    for(unsigned i=0; i<100; i++){

        try{

            likelihood_value = (*gl)(gp)[0];
            std::cout << "Iteration: " << i << ", likelihood: " << w*likelihood_value;

            VectorType likelihood_update = gl->GetParameterDerivatives(gp);
            //std::cout << ", gradient: " << update.transpose() << std::endl;

            if(likelihood_update.rows()!=parameters.rows()){
                std::cout << "[failed] Wrong number of parameters" << std::endl;
            }
            if(logpdfs.size() != parameters.rows()){
                std::cout << "[failed] wrong number of parameters or prior" << std::endl;
            }

            VectorType prior_update = VectorType::Zero(parameters.rows());
            for(unsigned p=0; p<parameters.rows(); p++){
                prior_update[p] = logpdfs[p].second->GetDerivative(parameters[p]);
            }

            prior_value = 0;
            for(unsigned p=0; p<pdfs.size(); p++){
                prior_value += (*pdfs[p].second)(parameters[p]);
            }

            for(unsigned p=0; p<parameters.rows(); p++){
                double gradient = w * likelihood_update[p] + (1-w) * prior_update[p];
                parameters[p] -= lambda * gradient;
            }

            std::cout << ", prior: " << (1-w) * prior_value << std::flush;

            std::cout << ", parameters: " << parameters.transpose() << std::flush;
            std::cout << ", likelihood_update: " << likelihood_update.transpose() << std::flush;
            std::cout << ", prior_update: " << prior_update.transpose() << std::endl;

            gp->SetKernel(GetKernel(parameters));
        }
        catch(std::string& s){
            std::cout << "Error: " << s << std::endl;
            break;
            //return;
        }
    }

    std::cout << "Parameters are: " << parameters.transpose() << std::endl;
    std::cout << "\"\"\"" << std::endl;
    //return;

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=80; i<3000; i++){
        VectorType x = VectorType::Zero(1); x[0] = i;
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(i);
    }

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

    std::stringstream ss;
    ss << "prior: " << (1-w) * prior_value << "\\n";
    ss << ", likelihood: " << w*likelihood_value << "\\n";
    ss << parameters.transpose();
    std::cout << "plt.title(\"" << ss.str() << "\")" << std::endl;
    std::cout << "plt.legend()" << std::endl;
    std::cout << "plt.show()" << std::endl;
}


int main (int argc, char *argv[]){
    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;
    std::cout << "\"\"\"" << std::endl;
    std::cout << "Gaussian likelihood kernel test: " << std::endl;
    try{
    Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
