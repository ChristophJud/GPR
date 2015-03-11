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
     * Test 1.1: perform maximum likelihood
     *      - try to infer a squared function
     *      - maximum is searched by brute force
     */
    std::cout << "Test 1: maximum log gaussian likelihood (brute force)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    //auto f = [](double x)->double { return 400*std::sin(1.5*x) + x*x; };
    auto f = [](double x)->double { return x*x; };

    double val = 0;
    unsigned n = 100;
    MatrixType signal = MatrixType::Zero(2,n);
    for(unsigned i=0; i<n; i++){
        signal(0,i) = val;
        signal(1,i) = f(val);
        val += 100.0/n;
    }

    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0.001); // zero since with the white kernel (see later) this is considered

    for(unsigned i=0; i<10; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }

    for(unsigned i=80; i<90; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }


    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());


    double sigma_max = 0;
    double scale_max = 0;
    double likelihood_max = std::numeric_limits<double>::lowest();

    for(double scale=1; scale<1000; scale+=10){
        for(double sigma=1; sigma<100; sigma+=1){
            // analytical
            try{
                GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
                gp->SetKernel(gk);

                double likelihood = (*gl)(gp)[0];
                if(likelihood > likelihood_max){
                    likelihood_max = likelihood;
                    sigma_max = sigma;
                    scale_max = scale;
                }
            }
            catch(std::string& s){
                std::cout << "[failed] " << s << std::endl;
                return;
            }
        }
    }

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=0; i<100; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(signal(0,i));
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 2){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }
}

void Test2(){
    /*
     * Test 2: perform maximum likelihood
     *      - try to infer a squared function
     *      - maximum is searched by brute force
     */
    std::cout << "Test 2: maximum log gaussian likelihood (gradient descent)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    //auto f = [](double x)->double { return 400*std::sin(1.5*x) + x*x; };
    auto f = [](double x)->double { return x*x; };

    double val = 0;
    unsigned n = 100;
    MatrixType signal = MatrixType::Zero(2,n);
    for(unsigned i=0; i<n; i++){
        signal(0,i) = val;
        signal(1,i) = f(val);
        val += 100.0/n;
    }

    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0.001); // zero since with the white kernel (see later) this is considered

    for(unsigned i=0; i<10; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }

    for(unsigned i=80; i<90; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }


    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());


    double sigma = 20;
    double scale = 100;
    double lambda = 1e-3;
    for(unsigned i=0; i<1000; i++){
        // analytical
        try{
            GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
            gp->SetKernel(gk);

            VectorType likelihood_update = gl->GetParameterDerivatives(gp);

            sigma -= lambda * likelihood_update[0];
            scale -= lambda * likelihood_update[1];
        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::endl;
            return;
        }
    }


    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=0; i<100; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(signal(0,i));
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 5){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }
}

void Test3(){
    /*
     * Test 3: perform maximum likelihood
     *      - try to infer a periodic function
     *      - maximum is searched by brute force
     */
    std::cout << "Test 3: maximum log gaussian likelihood (brute force)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    auto f = [](double x)->double { return 400*std::sin(1.5*x); };

    double val = 0;
    unsigned n = 200;
    MatrixType signal = MatrixType::Zero(2,n);
    for(unsigned i=0; i<n; i++){
        signal(0,i) = val;
        signal(1,i) = f(val);
        val += 50.0/n;
    }

    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0.001); // zero since with the white kernel (see later) this is considered

    for(unsigned i=0; i<20; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }

    for(unsigned i=80; i<100; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }


    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());


    //double period = M_PI/15;
    double sigma = 1;
    double scale = 400;
    double likelihood_max = std::numeric_limits<double>::lowest();
    double period_max = 0;
    for(double period=0; period<2*M_PI; period+=0.001){
        // analytical
        try{
            PeriodicKernelTypePointer gk(new PeriodicKernelType(scale, period, sigma));
            gp->SetKernel(gk);

            double likelihood= (*gl)(gp)[0];
            if(likelihood > likelihood_max){
                likelihood_max = likelihood;
                period_max = period;
            }
        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::endl;
            return;
        }
    }


    PeriodicKernelTypePointer gk(new PeriodicKernelType(scale, period_max, sigma));
    gp->SetKernel(gk);

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=0; i<signal.cols(); i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(signal(0,i));
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 1e-5){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }
}

int main (int argc, char *argv[]){
    std::cout << "Gaussian likelihood kernel test: " << std::endl;
    try{
        Test1();
        Test2();
        Test3();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
