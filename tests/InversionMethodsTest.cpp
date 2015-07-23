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

#include "GaussianProcess.h"
#include "Kernel.h"
#include "LAPACKUtils.h"

using namespace gpr;

typedef GaussianKernel<double>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

GaussianProcessTypePointer GetGaussianProcess(GaussianProcessType::InversionMethod method){
    KernelTypePointer k(new KernelType(2.8));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0);

    unsigned number_of_samples = 10;

    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = i * 2*M_PI/number_of_samples;

        VectorType y(1);
        y(0) = std::sin(x(0));
        gp->AddSample(x,y);
    }
    //gp->DebugOn();
    gp->SetInversionMethod(method);
    return gp;
}

double EvaluateGaussianProcess(GaussianProcessTypePointer gp){
    // perform prediction
    unsigned number_of_tests = 50;
    double err = 0;
    for(unsigned i=0; i<number_of_tests; i++){
        VectorType x(1);
        x(0) = i * 2*M_PI/number_of_tests;

        err += std::fabs(gp->Predict(x)(0)-std::sin(x(0)));
    }
    return err;
}

void Test1(){
    /*
     * Test 1: FullPivotLU test
     * - try to learn sinus function
     */
    std::cout << "Test 1: FullPivotLU...\t\t" << std::flush;

    GaussianProcessTypePointer gp = GetGaussianProcess(GaussianProcessType::FullPivotLU);
    gp->Initialize();
    double err = EvaluateGaussianProcess(gp);
    if(err>0.0006){
        std::cout << " [failed]. Prediction error: " << err << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }
}

void Test2(){
    /*
     * Test 2: JacobiSVD test
     * - try to learn sinus function
     */
    std::cout << "Test 2: JacobiSVD...\t\t" << std::flush;

    GaussianProcessTypePointer gp = GetGaussianProcess(GaussianProcessType::JacobiSVD);
    gp->Initialize();
    double err = EvaluateGaussianProcess(gp);
    if(err>0.0006){
        std::cout << " [failed]. Prediction error: " << err << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }
}

void Test3(){
    /*
     * Test 3: BDCSVD test
     * - try to learn sinus function
     */
    std::cout << "Test 3: BDCSVD...\t\t" << std::flush;

    try{
        GaussianProcessTypePointer gp = GetGaussianProcess(GaussianProcessType::BDCSVD);
        gp->Initialize();
        double err = EvaluateGaussianProcess(gp);
        if(err>0.0006){
            std::cout << " [failed]. Prediction error: " << err << std::endl;
        }
        else{
            std::cout << " [passed]." << std::endl;
        }
    }
    catch(std::string& s){
        std::cout << " [failed]. Because: " << s << std::endl;
    }
}

void Test4(){
    /*
     * Test 4: SelfAdjointEigenSolver test
     * - try to learn sinus function
     */
    std::cout << "Test 4: SelfAdjointEigenSolver..." << std::flush;

    GaussianProcessTypePointer gp = GetGaussianProcess(GaussianProcessType::SelfAdjointEigenSolver);
    gp->Initialize();
    double err = EvaluateGaussianProcess(gp);
    if(err>0.0006){
        std::cout << "[failed]. Prediction error: " << err << std::endl;
    }
    else{
        std::cout << "[passed]." << std::endl;
    }
}

void Test5(){
    std::cout << "Test 5: compare Eigen and Lapack inversion..." << std::flush;
    double err_lu = 0;
    double err_chol = 0;
    unsigned cnt = 0;
    for(unsigned i=0; i<20; i++){
        MatrixType M = MatrixType::Random(200,200);
        MatrixType MM = M*M.adjoint();
        MatrixType M_inv = M.inverse();
        MatrixType MM_inv = MM.inverse();
        MatrixType M_inv_lu = lapack::lu_invert<double>(M);
        MatrixType MM_inv_chol = lapack::chol_invert<double>(MM);

        err_lu += (M_inv - M_inv_lu).norm();
        err_chol += (MM_inv - MM_inv_chol).norm();

        cnt++;
    }

    if(err_lu/cnt > 1e-10){
        std::cout << " (LU) [failed] with an error of " << err_lu/cnt << std::endl;
        return;
    }
    if(err_chol/cnt > 1e-4){
        std::cout << " (cholesky) [failed] with an error of " << err_chol/cnt << std::endl;
        return;
    }
    std::cout << "[passed]" << std::endl;
}


int main (int argc, char *argv[]){
    std::cout << "Gaussian process regression with different inversion methods: " << std::endl;
    try{
        Test1();
        Test2();
        Test3();
        Test4();
        Test5();
    }
    catch(std::string& s){
        std::cout << "[failed] Error: " << s << std::endl;
    }

    return 0;
}

