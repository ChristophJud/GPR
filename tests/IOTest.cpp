
// compilation: g++ -I /home/jud/apps/Eigen_3.1.3/linux64/include/eigen3/ gp_main.cpp -o gpTest -std=c++0x

#include <iostream>
#include <memory>
#include <ctime>

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"
#include "MatrixIO.h"


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
        std::cout << "[passed]." << std::endl;
    }
    else{
        std::cout << "[failed]." << std::endl;
    }
}

void Test2(){
    /*
     * Test 2: gaussian process save and load
     */
    std::cout << "Test 2: save/load gaussian process... " << std::flush;
    KernelTypePointer k(new KernelType(1.8));
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
        std::cout << " [passed]." << std::endl;
    }
    else{
        std::cout << " [failed]." << std::endl;
    }
}


int main (int argc, char *argv[]){

    Test1();
    Test2();

    return 0;
}
