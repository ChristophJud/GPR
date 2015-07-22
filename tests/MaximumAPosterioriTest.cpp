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
    //std::cout << "Test 1: maximum log gaussian likelihood + log gaussian prior (gradient descent)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    //auto f = [](double x)->double { return std::sin(x)*std::cos(2.2*std::sin(x)); };
    auto f = [](double x)->double { return x+10*std::sin(x); };

    double val = 0;
    unsigned n = 70;
    double upper = 30;
    MatrixType signal = MatrixType::Zero(2,n);
    for(unsigned i=0; i<n; i++){
        signal(0,i) = val;
        signal(1,i) = f(val);
        val += upper/n;
    }

    // build Gaussian process and add the 1D samples
    WhiteKernelTypePointer pk(new WhiteKernelType(0)); // dummy kernel
    GaussianProcessTypePointer gp(new GaussianProcessType(pk));
    gp->SetSigma(0.01); // zero since with the white kernel (see later) this is considered

    for(unsigned i=0; i<n; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        VectorType y = VectorType::Zero(1); y[0] = signal(1,i);
        gp->AddSample(x,y);
    }

    bool cout = false;

    // scale/period/sigma: 18.525/3.00185/5.02189, sigma/scale: 16.8268/26.5342
//    double p_sigma = 3;
//    double p_sigma_variance = 0.1;
//    double p_scale = 20;
//    double p_scale_variance = 2;
//    double p_period = 3;
//    double p_period_variance = 0.1;
//    double g_sigma = 15;
//    double g_sigma_variance = 2;
//    double g_scale = 30;
//    double g_scale_variance = 2;

    double p_sigma = 3;
    double p_sigma_variance = 0.1;
    double p_scale = 20;
    double p_scale_variance = 2;
    double p_period = 3;
    double p_period_variance = 0.1;
    double g_sigma = 15;
    double g_sigma_variance = 2;
    double g_scale = 30;
    double g_scale_variance = 2;

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

    // construct Gaussian priors
    typedef gpr::GaussianDensity<double> GaussianDensityType;
    typedef std::shared_ptr<GaussianDensityType> GaussianDensityTypePointer;

//    std::vector<GaussianDensityTypePointer> g_densities;
//    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_scale, 0.2)));
//    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_period, 0.05)));
//    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(p_sigma, 2)));
//    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(g_sigma, 10)));
//    g_densities.push_back(GaussianDensityTypePointer(new GaussianDensityType(g_scale, 0.05)));


    typedef gpr::InverseGaussianDensity<long double> InverseGaussianDensityType;
    typedef std::shared_ptr<InverseGaussianDensityType> InverseGaussianDensityTypePointer;

    std::vector<InverseGaussianDensityTypePointer> g_densities;
    g_densities.push_back(InverseGaussianDensityTypePointer( new InverseGaussianDensityType(InverseGaussianDensityType::GetMeanAndLambda(p_scale, p_scale_variance))));
    g_densities.push_back(InverseGaussianDensityTypePointer( new InverseGaussianDensityType(InverseGaussianDensityType::GetMeanAndLambda(p_period, p_period_variance))));
    g_densities.push_back(InverseGaussianDensityTypePointer( new InverseGaussianDensityType(InverseGaussianDensityType::GetMeanAndLambda(p_sigma, p_sigma_variance))));
    g_densities.push_back(InverseGaussianDensityTypePointer( new InverseGaussianDensityType(InverseGaussianDensityType::GetMeanAndLambda(g_sigma, g_sigma_variance))));
    g_densities.push_back(InverseGaussianDensityTypePointer( new InverseGaussianDensityType(InverseGaussianDensityType::GetMeanAndLambda(g_scale, g_scale_variance))));

    if(cout){
        for(auto p : g_densities){
            std::cout << "mode : " << p->mode() << ", variance: " << p->variance() << std::endl;
        }
    }

    double lambda = 1e-3;
    double w = 0.3; // weighting of w*likelihood resp. (1-w)*prior
    for(unsigned i=0; i<1000; i++){
        // analytical
        try{
            PeriodicKernelTypePointer pk(new PeriodicKernelType(p_scale, M_PI/p_period, p_sigma));
            GaussianKernelTypePointer gk(new GaussianKernelType(g_sigma, g_scale));
            SumKernelTypePointer sk(new SumKernelType(pk,gk));
            gp->SetKernel(sk);

            GaussianLogLikelihoodType::ValueDerivativePair lp = gl->GetValueAndParameterDerivatives(gp);
            VectorType likelihood = lp.first;
            VectorType likelihood_gradient = lp.second;

            double prior = 0;
            prior += g_densities[0]->log(p_scale);
            prior += g_densities[1]->log(p_period);
            prior += g_densities[2]->log(p_sigma);
            prior += g_densities[3]->log(g_sigma);
            prior += g_densities[4]->log(g_scale);

            double posterior = -(w*likelihood[0]+(1-w)*prior);

            MatrixType J = MatrixType::Zero(1,g_densities.size());
            J(0,0) = w*likelihood_gradient[0] + (1-w)*g_densities[0]->GetLogDerivative(p_scale);
            J(0,1) = w*likelihood_gradient[1] + (1-w)*g_densities[1]->GetLogDerivative(p_period);
            J(0,2) = w*likelihood_gradient[2] + (1-w)*g_densities[2]->GetLogDerivative(p_sigma);
            J(0,3) = w*likelihood_gradient[3] + (1-w)*g_densities[3]->GetLogDerivative(g_sigma);
            J(0,4) = w*likelihood_gradient[4] + (1-w)*g_densities[4]->GetLogDerivative(g_scale);

            VectorType update = lambda * gpr::pinv<MatrixType>(J.adjoint()*J)*J.adjoint();

            if(cout){
                std::cout << i << ": likelihood: " << likelihood[0] << ", prior: " << prior << ", posterior: " << posterior;
                std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale;
                std::cout << ", update: " << update.adjoint() << std::endl;
                //std::cout << "log gauss: scale/period/sigma, sigma/scale: " << (*g_densities[0])(p_scale) << "/" << (*g_densities[1])(p_period) << "/" << (*g_densities[2])(p_sigma);
                //std::cout << ", " << (*g_densities[3])(g_sigma) << "/" << (*g_densities[4])(g_scale) << std::endl;
            }
//            p_scale += lambda * posterior / (likelihood_gradient[0] + g_densities[0]->GetLogDerivative(p_scale));
//            p_period += lambda * posterior / (likelihood_gradient[1] + g_densities[1]->GetLogDerivative(p_period));
//            p_sigma += lambda * posterior / (likelihood_gradient[2] + g_densities[2]->GetLogDerivative(p_sigma));
//            g_sigma += lambda * posterior / (likelihood_gradient[3] + g_densities[3]->GetLogDerivative(g_sigma));
//            g_scale += lambda * posterior / (likelihood_gradient[4] + g_densities[4]->GetLogDerivative(g_scale));

            p_scale -= update[0] * posterior;
            p_period -= update[1] * posterior;
            p_sigma -= update[2] * posterior;
            g_sigma -= update[3] * posterior;
            g_scale -= update[4] * posterior;

            if(p_scale < std::numeric_limits<long double>::min()) p_scale = std::numeric_limits<long double>::min();
            if(p_period < std::numeric_limits<long double>::min()) p_period = std::numeric_limits<long double>::min();
            if(p_sigma < std::numeric_limits<long double>::min()) p_sigma = std::numeric_limits<long double>::min();
            if(g_sigma < std::numeric_limits<long double>::min()) g_sigma = std::numeric_limits<long double>::min();
            if(g_scale < std::numeric_limits<long double>::min()) g_scale = std::numeric_limits<long double>::min();

        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::endl;
            std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
            return;
        }
    }

    std::cout << "print \"" << "scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << "\""<< std::endl;

    std::vector<double> prediction_y;
    std::vector<double> prediction_x;
    // predict some stuff:
    for(unsigned i=0; i<n; i++){
        VectorType x = VectorType::Zero(1); x[0] = signal(0,i);
        prediction_y.push_back(gp->Predict(x)[0]);
        prediction_x.push_back(signal(0,i));
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    std::cout << "print \"error: " << err/prediction_x.size() << "\"" << std::endl;

    if(cout) return;

    double N = 200;

    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;

    std::cout << "x = np.array([";
    for(unsigned i=0; i<N; i++){
        std::cout << i*upper/N << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "y = np.array([";
    for(unsigned i=0; i<N; i++){
        std::cout << f(i*upper/N) << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x,y)" << std::endl;


    std::cout << "xp = np.array([";
    for(unsigned i=0; i<N; i++){
        std::cout << i*upper/N << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "yp = np.array([";
    for(unsigned i=0; i<N; i++){
        std::cout << gp->Predict(VectorType::Constant(1,i*15/N))[0] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(xp,yp,)" << std::endl;


    std::cout << "X = np.array([";
    for(unsigned i=0; i<n; i++){
        std::cout << signal(0,i) << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "Y = np.array([";
    for(unsigned i=0; i<n; i++){
        std::cout << signal(1,i) << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(X,Y,'.k')" << std::endl;



    std::cout << "plt.show()" << std::endl;

    return;

    if(err/prediction_x.size() < 5){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed] with an error of " << err/prediction_x.size() << std::endl;
    }
}


int main (int argc, char *argv[]){
    //std::cout << "Gaussian likelihood kernel test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}

