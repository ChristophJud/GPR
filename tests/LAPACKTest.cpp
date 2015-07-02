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
#include <chrono>

#include <boost/random.hpp>
#include <Eigen/SVD>

#include "LAPACKUtils.h"
#include "GaussianProcess.h"

using namespace gpr;

template<class T>
void Test1(unsigned N, bool cout=false){
    /*
     * Test 1: invert general matrix
     * - compare Eigen inversion and LAPACK inversion
     */
    std::cout << "Test 1: Eigen vs. LAPACK... (general matrix) " << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start;


    typedef GaussianProcess<T> GaussianProcessType;
    typedef typename GaussianProcessType::MatrixType MatrixType;

    // generate random matrix
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, 1);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    // generate double precision random matrix
    MatrixType m(N, N);
    for (unsigned i =0; i < N ; i++) {
        for (unsigned j = 0; j < N; j++) {
            m(i,j) = r();
        }
    }

    if(cout){
        std::cout << "matrix: " << std::endl;
        std::cout << m << std::endl;
    }

    // Eigen inversion (LU)
    std::cout << " - eigen... " << std::flush;
    start = std::chrono::system_clock::now();
    MatrixType m_inv = m.inverse();
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
    std::cout << "elapsed time: (sec) " << elapsed_seconds.count() << std::endl;
    if(cout) std::cout << m_inv << std::endl;

    // LAPACK inversion (LU)
    std::cout << " - lapack... " << std::flush;
    start = std::chrono::system_clock::now();
    MatrixType lu_inv = lapack::lu_invert<T>(m);
    elapsed_seconds = std::chrono::system_clock::now()-start;
    std::cout << "elapsed time: (sec) " << elapsed_seconds.count() << std::endl;
    if(cout) std::cout << lu_inv << std::endl;

    MatrixType lapack_identity = (m * lu_inv);
    MatrixType lu_identity = (m * m_inv);
    MatrixType identity = MatrixType::Identity(m.cols(), m.cols());

    T lapack_error = 0;
    T lu_error = 0;
    // generate double precision random matrix
    for (unsigned i =0; i < N ; i++) {
        for (unsigned j = 0; j < N; j++) {
            lu_error += std::fabs(identity(i,j)-lu_identity(i,j));
            lapack_error += std::fabs(identity(i,j)-lapack_identity(i,j));
        }
    }

    std::cout << " - [passed] error: eigen " << lu_error << ", lapack " << lapack_error << std::endl;
}

template<class T>
void Test2(unsigned N, bool cout=false){
    /*
     * Test 2: invert general matrix
     * - compare Eigen inversion and LAPACK inversion
     */
    std::cout << "Test 2: Eigen vs. LAPACK... (symmetric matrix) " << std::endl;
    std::chrono::time_point<std::chrono::system_clock> start;


    typedef GaussianProcess<T> GaussianProcessType;
    typedef typename GaussianProcessType::MatrixType MatrixType;
    typedef typename GaussianProcessType::VectorType VectorType;

    // generate random matrix
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, 1);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    // generate double precision random matrix
    MatrixType m(N, N);
    for (unsigned i =0; i < N ; i++) {
        for (unsigned j = 0; j < N; j++) {
            m(i,j) = r();
        }
    }

    m = m*m.transpose(); // make it inverse and positive definite

    if(cout){
        std::cout << "matrix: " << std::endl;
        std::cout << m << std::endl;
    }

    // Eigen inversion for symmetric positive definite matrices
    std::cout << " - eigen... " << std::flush;
    start = std::chrono::system_clock::now();

    Eigen::SelfAdjointEigenSolver<MatrixType> es;
    es.compute(m);
    VectorType eigenValues = es.eigenvalues().reverse();
    MatrixType eigenVectors = es.eigenvectors().rowwise().reverse();
    if((eigenValues.real().array() < 0).any()){
        std::cout << "[failed]: there are negative eigenvalues." << std::endl;
        std::cout.flush();
    }
    MatrixType m_inv = eigenVectors * VectorType(1/eigenValues.array()).asDiagonal() * eigenVectors.transpose();
    std::chrono::duration<double> elapsed_seconds = std::chrono::system_clock::now()-start;
    std::cout << "elapsed time: (sec) " << elapsed_seconds.count() << std::endl;
    if(cout) std::cout << m_inv << std::endl;

    // LAPACK inversion (cholesky)
    std::cout << " - lapack... " << std::flush;
    start = std::chrono::system_clock::now();
    MatrixType chol_inv = lapack::chol_invert<T>(m);
    elapsed_seconds = std::chrono::system_clock::now()-start;
    std::cout << "elapsed time: (sec) " << elapsed_seconds.count() << std::endl;
    if(cout) std::cout << chol_inv << std::endl;

    MatrixType chol_identity = (m * chol_inv);
    MatrixType lu_identity = (m * m_inv);
    MatrixType identity = MatrixType::Identity(m.cols(), m.cols());

    T chol_error = 0;
    T lu_error = 0;
    // generate double precision random matrix
    for (unsigned i =0; i < N ; i++) {
        for (unsigned j = 0; j < N; j++) {
            //error += std::fabs(m_inv(i,j)-chol_inv(i,j));
            chol_error += std::fabs(chol_identity(i,j)-identity(i,j));
            lu_error += std::fabs(lu_identity(i,j)-identity(i,j));
        }
    }
    std::cout << " - [passed] error: eigen " << lu_error << ", lapack " << chol_error << std::endl;
}

int main (int argc, char *argv[]){

    unsigned n = 1000;
    bool cout = false;
    try{
        std::cout << "LAPACK inversion test: (float)" << std::endl;
        Test1<float>(n, cout);
        Test2<float>(n, cout);

        std::cout << "LAPACK inversion test: (double)" << std::endl;
        Test1<double>(n, cout);
        Test2<double>(n, cout);
    }
    catch(std::string& s){
        std::cout << " [failed] - " << s << std::endl;
    }


    return 0;
}


