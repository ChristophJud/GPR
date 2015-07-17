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


template<typename T>
void Test1(){
    /*
     * Test 1.1: perform regression
     * Test 1.2: save/load product kernel
     */
    std::cout << "Test 1.1: regression test with product kernel... " << std::flush;


    // typedefs
    typedef GaussianKernel<T>                   GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType> GaussianKernelTypePointer;
    typedef PeriodicKernel<T>                   PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType> PeriodicKernelTypePointer;
    typedef ProductKernel<T>                    ProductKernelType;
    typedef std::shared_ptr<ProductKernelType>  ProductKernelTypePointer;

    typedef GaussianProcess<T>                  GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType>GaussianProcessTypePointer;

    typedef typename GaussianProcessType::VectorType     VectorType;
    typedef typename GaussianProcessType::MatrixType     MatrixType;


    // ground truth function
    auto f = [](double x)->double { return x/2.0 * std::sin(x)*std::cos(2.2*std::sin(x)); };

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
    double noise = 0.04;
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double interval_training_end = 3 * 2*M_PI; // interval to train
    unsigned number_of_samples = 50;

    PeriodicKernelTypePointer   pk(new PeriodicKernelType(5, 0.5, 0.4));
    GaussianKernelTypePointer   gk(new GaussianKernelType(65, 10));
    ProductKernelTypePointer    cpk(new ProductKernelType(pk, gk));

    GaussianProcessTypePointer gp(new GaussianProcessType(cpk));
    gp->SetSigma(noise);

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


    double err = (y-y_predict).norm() / gt_size;
    if(err>0.02){
        std::cout << " [failed]. Prediction error: " << err << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }

    std::cout << "Test 1.2: save/load product kernel... " << std::flush;

    gp->Save("/tmp/gp_io_test-");


    GaussianKernelTypePointer k_dummy(new GaussianKernelType(1, 1));
    GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));
    gp_read->Load("/tmp/gp_io_test-");

    if(*gp.get() == *gp_read.get()){
        std::cout << " [passed]." << std::endl;
    }
    else{
        std::cout << " [failed]." << std::endl;
    }
}


template<typename T>
void Test2(){
    // typedefs
    typedef GaussianKernel<T>                   GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>         GaussianKernelTypePointer;
    typedef PeriodicKernel<T>                   PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType>         PeriodicKernelTypePointer;
    typedef ProductKernel<T>                        ProductKernelType;
    typedef std::shared_ptr<ProductKernelType>      ProductKernelTypePointer;


    PeriodicKernelTypePointer   pk(new PeriodicKernelType(0.59, 0.5, 0.4));
    GaussianKernelTypePointer   gk(new GaussianKernelType(132, M_PI));
    ProductKernelTypePointer        sk(new ProductKernelType(pk, gk));

    ProductKernelTypePointer sk2(new ProductKernelType(PeriodicKernelTypePointer(new PeriodicKernelType(1, 1, 1)),
                                               GaussianKernelTypePointer(new GaussianKernelType(1, 1))));

    sk2->SetParameters(sk->GetParameters());

    if((*sk) != (*sk2)){
        std::cout << " [failed]." << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }

}

int main (int argc, char *argv[]){

    try{
        std::cout << "Test 1: Product kernel test (float): " << std::endl;
        Test1<float>();
        std::cout << "Test 1: Product kernel test (double): " << std::endl;
        Test1<double>();

        std::cout << "Test 2: parameter test with product kernel (float) " << std::flush;
        Test2<float>();
        std::cout << "Test 2: parameter test with product kernel (double) " << std::flush;
        Test2<double>();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}


