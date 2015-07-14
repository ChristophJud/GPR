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

typedef RationalQuadraticKernel<double> RationalQuadraticKernelType;
typedef std::shared_ptr<RationalQuadraticKernelType> RationalQuadraticKernelTypePointer;
typedef GaussianKernel<double> GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType> GaussianKernelTypePointer;
typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;


void Test1(){
    /*
     * Test 1: regression test 1D
     */
    std::cout << "Test 1: equal to gaussian kernel with large alpha test..." << std::flush;

    // ground truth periodic variable
    auto f = [](double x)->double { return 0.5*x*std::sin(x)+std::sin(4*x); };

    double interval_start = -10;
    double interval_end = 10; // full interval
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

    double interval_training_end = 5; // interval to train
    unsigned number_of_samples = 50;

    // kernels
    RationalQuadraticKernelTypePointer rq(new RationalQuadraticKernelType(4, 1, 1e10)); // scale, period, smoothness
    GaussianKernelTypePointer gk(new GaussianKernelType(1,4));

    // gps
    GaussianProcessTypePointer rq_gp(new GaussianProcessType(rq));
    rq_gp->SetSigma(noise); // noise
    GaussianProcessTypePointer gk_gp(new GaussianProcessType(gk));
    gk_gp->SetSigma(noise); // noise

    // add samples
    double training_step_size = (interval_training_end - interval_start) / number_of_samples;
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = interval_start + i*training_step_size;

        VectorType y(1);
        y(0) = f(x(0)) + r();

        rq_gp->AddSample(x, y);
        gk_gp->AddSample(x, y);
    }
    rq_gp->Initialize();
    gk_gp->Initialize();

    //--------------------------------------------------------------------------------
    // predict full intervall
    VectorType y_predict_rq(gt_size);
    VectorType y_predict_gk(gt_size);
    for(unsigned i=0; i<gt_size; i++){
        VectorType x(1);
        x(0) = interval_start + i*interval_step;
        y_predict_rq[i] = rq_gp->Predict(x)(0);
        y_predict_gk[i] = gk_gp->Predict(x)(0);
    }

    double err = (y_predict_gk-y_predict_rq).norm();
    if(err>0.01){
        std::cout << " [failed]." << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }

}

void Test2(){
    /*
     * Test 2: regression test 1D
     */
    std::cout << "Test 2: perform standard regression..." << std::flush;

    // ground truth periodic variable
    auto f = [](double x)->double { return 0.5*x*std::sin(x)+std::sin(4*x); };

    double interval_start = -10;
    double interval_end = 10; // full interval
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
    double noise = 0.01;
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double interval_training_end = 10; // interval to train
    unsigned number_of_samples = 50;

    // kernels
    RationalQuadraticKernelTypePointer rq(new RationalQuadraticKernelType(4, 2.5, 0.01)); // scale, period, smoothness

    // gps
    GaussianProcessTypePointer rq_gp(new GaussianProcessType(rq));
    rq_gp->SetSigma(noise); // noise

    // add samples
    double training_step_size = (interval_training_end - interval_start) / number_of_samples;
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = interval_start + i*training_step_size;

        VectorType y(1);
        y(0) = f(x(0)) + r();

        rq_gp->AddSample(x, y);
    }
    rq_gp->Initialize();


    //--------------------------------------------------------------------------------
    // predict full intervall
    VectorType y_predict_rq(gt_size);
    for(unsigned i=0; i<gt_size; i++){
        VectorType x(1);
        x(0) = interval_start + i*interval_step;
        y_predict_rq[i] = rq_gp->Predict(x)(0);
    }


    double err = (y-y_predict_rq).norm();
    if(err>3){
        std::cout << " [failed]." << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }

}

int main (int argc, char *argv[]){
    std::cout << "Rational quadratic kernel test: " << std::endl;
    Test1();
    Test2();

    return 0;
}


