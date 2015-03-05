
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

typedef Kernel<double>                      KernelType;
typedef std::shared_ptr<KernelType>         KernelTypePointer;
typedef GaussianKernel<double>              GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType> GaussianKernelTypePointer;
typedef WhiteKernel<double>                 WhiteKernelType;
typedef std::shared_ptr<WhiteKernelType>    WhiteKernelTypePointer;
typedef PeriodicKernel<double>              PeriodicKernelType;
typedef std::shared_ptr<PeriodicKernelType> PeriodicKernelTypePointer;
typedef RationalQuadraticKernel<double>     RationalQuadraticKernelType;
typedef std::shared_ptr<RationalQuadraticKernelType> RationalQuadraticKernelTypePointer;
typedef SumKernel<double>                   SumKernelType;
typedef std::shared_ptr<SumKernelType>      SumKernelTypePointer;
typedef ProductKernel<double>               ProductKernelType;
typedef std::shared_ptr<ProductKernelType>  ProductKernelTypePointer;

typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;


/*
 *  p0 = scale of k1
 *  p1 = sigma of k1
 *  p2 = gauss scale of k2
 *  p3 = gauss sigma of k2
 *  p4 = period scale of k2
 *  p5 = period period of k2
 *  p6 = period sigma of k2
 *  p7 = rational scale of k3
 *  p8 = rational sigma of k3
 *  p9 = rational alpha of k3
 *  p10 = gauss scale of k4
 *  p11 = gauss sigma of k4
 *  p12 = white scale of k4
 *
 */
KernelTypePointer GetKernel(const std::vector<double>& params){
    if(params.size() != 13){
        throw std::string("Wrong number of arguments.");
    }

    GaussianKernelTypePointer k1(new GaussianKernelType(params[1],params[0]));

    GaussianKernelTypePointer k2_gk(new GaussianKernelType(params[3],params[2]));
    PeriodicKernelTypePointer k2_pk(new PeriodicKernelType(params[4],params[5],params[6]));
    ProductKernelTypePointer k2(new ProductKernelType(k2_gk, k2_pk));

    RationalQuadraticKernelTypePointer k3(new RationalQuadraticKernelType(params[7],params[8],params[9]));

    GaussianKernelTypePointer k4_gk(new GaussianKernelType(params[11],params[10]));
    WhiteKernelTypePointer k4_wk(new WhiteKernelType(params[12]));
    SumKernelTypePointer k4(new SumKernelType(k4_gk, k4_wk));

    SumKernelTypePointer k12(new SumKernelType(k1, k2));
    SumKernelTypePointer k123(new SumKernelType(k12,k3));
    SumKernelTypePointer k1234(new SumKernelType(k123,k4));

    return k1234;
}


void Test1(){
    /*
     * Test 1: construct, regress, io of highly general kernel
     */
    std::cout << "Test 1.1: construct, regress, of highly general kernel..." << std::flush;

    // ground truth periodic variable
    auto f = [](double x)->double { return std::sin(x)*std::cos(2.2*std::sin(x)); };

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
    double noise = 0.01;
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double interval_training_end = 2 * 2*M_PI; // interval to train
    unsigned number_of_samples = 50;

    std::vector<double> params;
    params.push_back(1);
    params.push_back(2);
    params.push_back(3);
    params.push_back(4);
    params.push_back(5);
    params.push_back(6);
    params.push_back(7);
    params.push_back(8);
    params.push_back(9);
    params.push_back(101);
    params.push_back(202);
    params.push_back(303);
    params.push_back(404);

    KernelTypePointer k = GetKernel(params);
    std::string k_string = k->ToString();

    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0); // noise

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

    //--------------------------------------------------------------------------------
    // predict full intervall
    VectorType y_predict(gt_size);
    for(unsigned i=0; i<gt_size; i++){
        VectorType x(1);
        x(0) = interval_start + i*interval_step;
        y_predict[i] = gp->Predict(x)(0);
    }

    double err = (y-y_predict).norm(); // here we are not interested in an accurate regression
    if(err>6){
        std::cout << "\t[failed]." << std::endl;
    }
    else{
        std::cout << "\t[passed]." << std::endl;
    }

    std::cout << "Test 1.2: save/load of highly general kernel..." << std::flush;

    gp->Save("/tmp/gp_io_test-");


    WhiteKernelTypePointer k_dummy(new WhiteKernelType(1));
    GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));


    try{
        gp_read->Load("/tmp/gp_io_test-");
    }
    catch(std::string& s){
        std::cout << s << std::endl;
    }


    std::string k_string_read = gp_read->GetKernel()->ToString();

    if(*gp.get() == *gp_read.get() && k_string_read.compare(k_string)==0){
        std::cout << "\t\t\t[passed]." << std::endl;
    }
    else{
        std::cout << "\t\t\t[failed]." << std::endl;
    }
}




int main (int argc, char *argv[]){
    std::cout << "Highly general kernel test: " << std::endl;
    Test1();

    return 0;
}

