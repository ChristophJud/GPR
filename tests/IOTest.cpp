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

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"
#include "MatrixIO.h"

using namespace gpr;

typedef GaussianKernel<double>		DPKernelType;
typedef std::shared_ptr<DPKernelType> DPKernelTypePointer;
typedef GaussianProcess<double> DPGaussianProcessType;
typedef std::shared_ptr<DPGaussianProcessType> DPGaussianProcessTypePointer;

typedef GaussianKernel<float>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<float> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef DPGaussianProcessType::VectorType DPVectorType;
typedef DPGaussianProcessType::MatrixType DPMatrixType;
typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

void Test1(){
    /*
     * Test 1: write and read matrix
     * - generate random matrix and look if the Write/Read function works
     */
    std::cout << "Test 1: write/read random matrix ..." << std::flush;

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, 1);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    unsigned N = 100;
    unsigned K = 50;

    // generate double precision random matrix
    DPMatrixType dp_m(N, K);
    for (unsigned i =0; i < N ; i++) {
        for (unsigned j = 0; j < K ; j++) {
            dp_m(i,j) = r();
        }
    }
    WriteMatrix<DPMatrixType>(dp_m, "/tmp/gpr_matrix_io_test_dp.txt");
    DPMatrixType dp_m_read = ReadMatrix<DPMatrixType>("/tmp/gpr_matrix_io_test_dp.txt");

    double dp_err = (dp_m - dp_m_read).norm();

    // generate double precision random matrix
    MatrixType sp_m(K, N);
    for (unsigned i =0; i < K ; i++) {
        for (unsigned j = 0; j < N ; j++) {
            sp_m(i,j) = r();
        }
    }

    WriteMatrix<MatrixType>(sp_m, "/tmp/gpr_matrix_io_test_sp.txt");
    MatrixType sp_m_read = ReadMatrix<MatrixType>("/tmp/gpr_matrix_io_test_sp.txt");

    double sp_err = (sp_m - sp_m_read).norm();

    if(sp_err == 0 && dp_err == 0){
        std::cout << "\t\t\t [passed]." << std::endl;
    }
    else{
        std::stringstream ss; ss<<sp_err <<", "<< dp_err; throw ss.str();
    }
}

void Test2(){
    /*
     * Test 2: gaussian process save and load
     */
    std::cout << "Test 2: save/load gaussian process... " << std::flush;
    KernelTypePointer k(new KernelType(std::sqrt(2)));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0);

    unsigned number_of_samples = 10;

    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(2);
        x(0) = x(1) = i * 2*M_PI/number_of_samples;

        VectorType y(2);
        y(0) = std::sin(x(0));
        y(1) = std::cos(x(1));

        gp->AddSample(x,y);
    }
    gp->Initialize();

    gp->Save("/tmp/gp_io_test-");


    KernelTypePointer k_dummy(new KernelType(1));
    GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));
    gp_read->Load("/tmp/gp_io_test-");

    if(*gp.get() == *gp_read.get()){
        std::cout << "\t\t\t [passed]." << std::endl;
    }
    else{
        throw std::string("read/write");
    }
}

void Test3(){
    /*
     * Test 3: gaussian process save and load with efficient setting
     *         (core matrix is not saved)
     */
    std::cout << "Test 3.1: (efficient) save/load gaussian process... " << std::flush;
    KernelTypePointer k(new KernelType(std::sqrt(2)));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0);
    if(gp->GetEfficientStorage()){
        std::cout << "\t [failed]. Efficient storage setting has to be turned off by default." << std::endl;
        return;
    }
    gp->SetEfficientStorage(true);

    unsigned number_of_samples = 10;

    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(2);
        x(0) = x(1) = i * 2*M_PI/number_of_samples;

        VectorType y(2);
        y(0) = std::sin(x(0));
        y(1) = std::cos(x(1));

        gp->AddSample(x,y);
    }
    gp->Initialize();

    // predict a point to compare later
    VectorType x(2);
    x(0) = x(1) = 2.56 * 2*M_PI/number_of_samples;

    VectorType yt(2);
    yt= gp->Predict(x);

    gp->Save("/tmp/gp_io_test-");

    {
        KernelTypePointer k_dummy(new KernelType(1));
        GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));
        gp_read->Load("/tmp/gp_io_test-");

        if(*gp.get() == *gp_read.get() && (yt-gp_read->Predict(x)).norm() < 1e-06){
            std::cout << "\t [passed]." << std::endl;
        }
        else{
            std::stringstream ss; ss<<(yt-gp_read->Predict(x)).norm(); throw ss.str();
        }
    }


    std::cout << "Test 3.2: (efficient) save/load gaussian process... " << std::flush;
    {
        // testing if compare operator works find with the core matrix
        KernelTypePointer k_dummy(new KernelType(1));
        GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));
        gp_read->Load("/tmp/gp_io_test-");

        float c = (*gp_read)(x, x); // core matrix should be built
        if(*gp.get() == *gp_read.get()){
            throw std::string("Comparison with late core matrix construction not right.");
            return;
        }
        c = (*gp)(x, x); // now the core matrix of gp should be built as well
        if(*gp.get() != *gp_read.get()){
            throw std::string("Comparison with late core matrix construction not right.");
            return;
        }

        if((yt-gp_read->Predict(x)).norm() < 1e-06){
            std::cout << " \t [passed]." << std::endl;
        }
        else{
            std::stringstream ss; ss<<(yt-gp_read->Predict(x)).norm(); throw ss.str();
        }
    }

    std::cout << "Test 3.3: (efficient) save/load gaussian process... " << std::flush;
    {
        float c = (*gp)(x, x); // ensure that core matrix is built
        gp->Save("/tmp/gp_io_test-");


        KernelTypePointer k_dummy(new KernelType(1));
        GaussianProcessTypePointer gp_read(new GaussianProcessType(k_dummy));
        gp_read->Load("/tmp/gp_io_test-");

        if(*gp.get() == *gp_read.get() && (yt-gp_read->Predict(x)).norm() < 1e-06){
            std::cout << " \t [passed]." << std::endl;
        }
        else{
            std::stringstream ss; ss<<(yt-gp_read->Predict(x)).norm(); throw ss.str();
        }
    }
}


int main (int argc, char *argv[]){
    std::cout << "Input/Output test:" << std::endl;
    try{
        Test1();
        Test2();
        Test3();
    }
    catch(std::string& s){
        std::cout << "[failed] Error: " << s << std::endl;
        return -1;
    }

    return 0;
}
