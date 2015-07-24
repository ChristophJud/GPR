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

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"
#include "MatrixIO.h"

using namespace gpr;

typedef GaussianProcess<float>                  GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType>    GaussianProcessTypePointer;

typedef typename GaussianProcessType::VectorType     VectorType;
typedef typename GaussianProcessType::MatrixType     MatrixType;

// setup gaussian process and assign it to the gp_out argument
GaussianProcessTypePointer BuildGaussianProcess(){
    // typedefs
    typedef GaussianKernel<float>                   GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType> GaussianKernelTypePointer;
    typedef PeriodicKernel<float>                   PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType> PeriodicKernelTypePointer;
    typedef SumKernel<float>                        SumKernelType;
    typedef std::shared_ptr<SumKernelType>      SumKernelTypePointer;

    // ground truth function
    auto f = [](double x)->double { return x/2.0 + std::sin(x)*std::cos(2.2*std::sin(x)); };

    double interval_start = 0;
    double interval_end = 5 * 2*M_PI; // full interval
    double interval_step = 0.1;

    //--------------------------------------------------------------------------------
    // generating ground truth
    unsigned gt_size = (interval_end-interval_start) / interval_step;
    VectorType y(gt_size);
    for(unsigned i=0; i<gt_size; i++){
        y[i] = f(interval_start + i*interval_step);
    }


    //--------------------------------------------------------------------------------
    // perform training
    double noise = std::sqrt(0.01);
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double interval_training_end = 2 * 2*M_PI; // interval to train
    unsigned number_of_samples = 50;

    PeriodicKernelTypePointer   pk(new PeriodicKernelType(0.59, 0.5, 0.4));
    GaussianKernelTypePointer   gk(new GaussianKernelType(130.0, M_PI));
    SumKernelTypePointer        sk(new SumKernelType(pk, gk));

    GaussianProcessTypePointer gp(new GaussianProcessType(sk));
    gp->SetSigma(std::sqrt(0.001));

    // add samples
    double training_step_size = (interval_training_end - interval_start) / number_of_samples;
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = interval_start + i*training_step_size;

        VectorType y(1);
        y(0) = f(x(0)) + r();

        gp->AddSample(x, y);
    }

    gp->Initialize();

    return gp;
}


void Test1(){
    /*
     * Test 1: scope test to show if everything with memory behaves well.
     */
    std::cout << "Test 1: scope test... " << std::flush;

    GaussianProcessTypePointer gp = BuildGaussianProcess();

    double interval_start = 0;
    double interval_end = 5 * 2*M_PI; // full interval
    double interval_step = 0.1;

    //--------------------------------------------------------------------------------
    // generating ground truth
    unsigned gt_size = (interval_end-interval_start) / interval_step;
    auto f = [](double x)->double { return x/2.0 + std::sin(x)*std::cos(2.2*std::sin(x)); };

    //--------------------------------------------------------------------------------
    // do some prediction
    VectorType y_predict(gt_size);
    VectorType y(gt_size);
    for(unsigned i=0; i<gt_size; i++){
        VectorType x(1);
        x(0) = interval_start + i*interval_step;
        y_predict[i] = gp->Predict(x)(0);
        y[i] = f(x(0));
    }


    double err = (y-y_predict).norm();
    if(err>3){
        std::stringstream ss; ss<<err; throw ss.str();
    }
    else{
        std::cout << " [passed]." << std::endl;
    }
}


int main (int argc, char *argv[]){
    std::cout << "Scope test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "[failed] Error: " << s << std::endl;
        return -1;
    }

    return 0;
}


