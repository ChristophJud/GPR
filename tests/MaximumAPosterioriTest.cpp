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

    double p_sigma = 1;
    double p_sigma_variance = 1;
    double p_scale = 10;
    double p_scale_variance = 1;
    double p_period = 6.3;
    double p_period_variance = 2;
    double g_sigma = 100;
    double g_sigma_variance = 20;
    double g_scale = 60;
    double g_scale_variance = 30;

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

    // construct Gaussian priors
    typedef gpr::GaussianDensity<double> GaussianDensityType;
    typedef std::shared_ptr<GaussianDensityType> GaussianDensityTypePointer;

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

    double lambda = 1e-6;
    double w = 0.8; // weighting of w*likelihood resp. (1-w)*prior
    for(unsigned i=0; i<100; i++){
        // analytical
        try{
            PeriodicKernelTypePointer pk(new PeriodicKernelType(p_scale, M_PI/p_period, p_sigma));
            GaussianKernelTypePointer gk(new GaussianKernelType(g_sigma, g_scale));
            SumKernelTypePointer sk(new SumKernelType(pk,gk));
            gp->SetKernel(sk);
            gp->SetSigma(0.1);
            //gp->DebugOn();

            GaussianLogLikelihoodType::ValueDerivativePair lp = gl->GetValueAndParameterDerivatives(gp);
            VectorType likelihood = lp.first;
            VectorType likelihood_gradient = lp.second;

            double prior = 0;
            prior += g_densities[0]->log(p_scale);
            prior += g_densities[1]->log(p_period);
            prior += g_densities[2]->log(p_sigma);
            prior += g_densities[3]->log(g_sigma);
            prior += g_densities[4]->log(g_scale);

            double posterior = -2*(w*likelihood[0]+(1-w)*prior);

            MatrixType J = MatrixType::Zero(1,g_densities.size());
            //std::cout << likelihood_gradient << std::endl;
            J(0,0) = 2*(w*likelihood_gradient[0] + (1-w)*g_densities[0]->GetLogDerivative(p_scale));
            J(0,1) = 2*(w*likelihood_gradient[1] + (1-w)*g_densities[1]->GetLogDerivative(p_period));
            J(0,2) = 2*(w*likelihood_gradient[2] + (1-w)*g_densities[2]->GetLogDerivative(p_sigma));
            J(0,3) = 2*(w*likelihood_gradient[3] + (1-w)*g_densities[3]->GetLogDerivative(g_sigma));
            J(0,4) = 2*(w*likelihood_gradient[4] + (1-w)*g_densities[4]->GetLogDerivative(g_scale));

            VectorType update = lambda * gpr::pinv<MatrixType>(J.adjoint()*J)*J.adjoint();

            if(cout){
                std::cout << i << ": likelihood: " << likelihood[0] << ", prior: " << prior << ", posterior: " << posterior;
                std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale;
                std::cout << ", update: " << update.adjoint() << std::endl;
            }

            p_scale -= update[0] * posterior;
            p_period -= update[1] * posterior;
            p_sigma -= update[2] * posterior;
            g_sigma -= update[3] * posterior;
            g_scale -= update[4] * posterior;

            if(p_scale < std::numeric_limits<double>::min()) p_scale = std::numeric_limits<double>::min();
            if(p_period < std::numeric_limits<double>::min()) p_period = std::numeric_limits<double>::min();
            if(p_sigma < std::numeric_limits<double>::min()) p_sigma = std::numeric_limits<double>::min();
            if(g_sigma < std::numeric_limits<double>::min()) g_sigma = std::numeric_limits<double>::min();
            if(g_scale < std::numeric_limits<double>::min()) g_scale = std::numeric_limits<double>::min();

        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::flush;
            std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
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
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 0.5){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed] with an error of " << err/prediction_x.size() << std::endl;
    }
}

void Test2(){
    /*
     * Test 1: perform maximum aposteriori
     */
    std::cout << "Test 2: maximum log gaussian likelihood + log gaussian prior (gradient descent, only period length)... " << std::flush;

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

    double p_sigma = 1;
    double p_sigma_variance = 1;
    double p_scale = 10;
    double p_scale_variance = 1;
    double p_period = 5;// should be 6.3
    double p_period_variance = 2;
    double g_sigma = 100;
    double g_sigma_variance = 20;
    double g_scale = 60;
    double g_scale_variance = 30;

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

    // construct Gaussian priors
    typedef gpr::GaussianDensity<double> GaussianDensityType;
    typedef std::shared_ptr<GaussianDensityType> GaussianDensityTypePointer;

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

    double w = 0.5; // weighting of w*likelihood resp. (1-w)*prior
    for(unsigned i=0; i<100; i++){
        // analytical
        try{
            PeriodicKernelTypePointer pk(new PeriodicKernelType(p_scale, M_PI/p_period, p_sigma));
            GaussianKernelTypePointer gk(new GaussianKernelType(g_sigma, g_scale));
            SumKernelTypePointer sk(new SumKernelType(pk,gk));
            gp->SetKernel(sk);
            gp->SetSigma(0.1);
            //gp->DebugOn();

            GaussianLogLikelihoodType::ValueDerivativePair lp = gl->GetValueAndParameterDerivatives(gp);
            VectorType likelihood = lp.first;
            VectorType likelihood_gradient = lp.second;

            double prior = 0;
            prior += g_densities[0]->log(p_scale);
            prior += g_densities[1]->log(p_period);
            prior += g_densities[2]->log(p_sigma);
            prior += g_densities[3]->log(g_sigma);
            prior += g_densities[4]->log(g_scale);

            double posterior = -2*(w*likelihood[0]+(1-w)*prior);


            double p_dev = 2*(w*likelihood_gradient[1] + (1-w)*g_densities[1]->GetLogDerivative(p_period));

            if(cout){
                std::cout << i << ": likelihood: " << likelihood[0] << ", prior: " << prior << ", posterior: " << posterior;
                std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
            }
            p_period -= posterior/p_dev;

            if(p_period < std::numeric_limits<double>::min()) p_period = std::numeric_limits<double>::min();

        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::endl;
            std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
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
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }

    if(err/prediction_x.size() < 0.1){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed] with an error of " << err/prediction_x.size() << std::endl;
    }
}

void Test3(){
    /*
     * Test 3: perform maximum aposteriori
     */
    std::cout << "Test 3: maximum log gaussian likelihood + log gaussian prior (gradient descent only sigma)... " << std::flush;

    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;

    // generating a signal
    // ground truth periodic variable
    //auto f = [](double x)->double { return std::sin(x)*std::cos(2.2*std::sin(x)); };
    auto f = [](double x)->double { return x+10*std::sin(x); };

    double val = 0;
    unsigned n = 30;
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

    double p_sigma = 0.5;
    double p_sigma_variance = 3;
    double p_scale = 10;
    double p_scale_variance = 1;
    double p_period = 6.3;
    double p_period_variance = 2;
    double g_sigma = 100;
    double g_sigma_variance = 20;
    double g_scale = 60;
    double g_scale_variance = 30;

    // construct Gaussian log likelihood
    typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
    typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
    GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

    // construct Gaussian priors
    typedef gpr::GaussianDensity<double> GaussianDensityType;
    typedef std::shared_ptr<GaussianDensityType> GaussianDensityTypePointer;

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

    double lambda = 1e-2;
    double w = 0.9; // weighting of w*likelihood resp. (1-w)*prior
    for(unsigned i=0; i<500; i++){
        // analytical
        try{
            PeriodicKernelTypePointer pk(new PeriodicKernelType(p_scale, M_PI/p_period, p_sigma));
            GaussianKernelTypePointer gk(new GaussianKernelType(g_sigma, g_scale));
            SumKernelTypePointer sk(new SumKernelType(pk,gk));
            gp->SetKernel(sk);
            gp->SetSigma(0.1);
            //gp->DebugOn();

            GaussianLogLikelihoodType::ValueDerivativePair lp = gl->GetValueAndParameterDerivatives(gp);
            VectorType likelihood = lp.first;
            VectorType likelihood_gradient = lp.second;

            double prior = 0;
            prior += g_densities[0]->log(p_scale);
            prior += g_densities[1]->log(p_period);
            prior += g_densities[2]->log(p_sigma);
            prior += g_densities[3]->log(g_sigma);
            prior += g_densities[4]->log(g_scale);

            double posterior = -2*(w*likelihood[0]+(1-w)*prior);

            double p_dev = g_densities[2]->GetLogDerivative(p_sigma);

            if(p_dev==0) break;

            double update;
            if(posterior/p_dev>0){
                update = std::log(1+posterior/p_dev);
            }
            else{
                update = -std::log(1+std::fabs(posterior/p_dev));
            }

            if(cout){
                std::cout << i << ": likelihood: " << likelihood[0] << ", prior: " << prior << ", posterior: " << posterior;
                std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale;
                std::cout << ", gradient p_sigma: " << p_dev << ", update " << update << std::endl;
            }
            p_sigma -= lambda*update;

            if(p_sigma < std::numeric_limits<double>::min()) p_sigma = std::numeric_limits<double>::min();

        }
        catch(std::string& s){
            std::cout << "[failed] " << s << std::flush;
            std::cout << ", scale/period/sigma: " << p_scale << "/" << p_period << "/" << p_sigma << ", sigma/scale: " << g_sigma << "/" << g_scale << std::endl;
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
    }

    double err = 0;
    for(unsigned i=0; i<prediction_x.size(); i++ ){
        err += std::abs(prediction_y[i]-signal(1,i));
    }


    if(err/prediction_x.size() < 0.5){
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
        Test2();
        Test3();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}

