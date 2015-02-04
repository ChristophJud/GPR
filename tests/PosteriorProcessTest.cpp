// compilation: g++ -I /home/jud/apps/Eigen_3.1.3/linux64/include/eigen3/ gp_main.cpp -o gpTest -std=c++0x

#include <iostream>
#include <memory>
#include <ctime>

#include <boost/random.hpp>

#include <Eigen/SVD>

#include "GaussianProcess.h"
#include "Kernel.h"

using namespace gpr;

typedef GaussianKernel<double>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<double> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;

VectorType GetRandomVector(unsigned n){
    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, 1);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    VectorType v = VectorType::Zero(n);
    for (unsigned i=0; i < n; i++) {
        v(i) = r();
    }
    return v;
}

void Test1(){
    /*
     * Test 1: scalar valued GP
     * - try to learn sinus function
     */
    std::cout << "Test 1: sinus regression... " << std::flush;

    KernelTypePointer k(new KernelType(0.5));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0.00001);

    unsigned number_of_samples = 20;

    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = i * 2*M_PI/number_of_samples;

        VectorType y(1);
        y(0) = std::sin(x(0));
        gp->AddSample(x,y);
    }
    gp->Initialize();


    // calculate credible interval
    unsigned number_of_tests = 50;
    for(unsigned i=0; i<number_of_tests; i++){
        VectorType x(1);
        x(0) = i * 2*M_PI/number_of_tests * 1.3;

        try{
            double c = 2*std::sqrt((*gp)(x,x)) - gp->GetCredibleInterval(x);
            if(c != 0){
                std::cout << " [failed] credible interval not correct." << std::endl;
                return;
            }
        }
        catch(std::string& s){
            std::cout << " [failed] error in calculating credible interval." << std::endl;
            return;
        }
    }
    std::cout << " [passed]" << std::endl;
}

void Test2(){
    /*
     * Test 2: sampling test
     * - try to sample from a posterior
     */
    std::cout << "Test 2: posterior sampling test... " << std::flush;

    KernelTypePointer k(new KernelType(1));
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(0);

    // add some landmarks
    gp->AddSample(VectorType::Ones(1),  VectorType::Zero(1));
    gp->AddSample(VectorType::Ones(1)*2,VectorType::Ones(1));
    gp->AddSample(VectorType::Ones(1)*3,VectorType::Ones(1)*0.5);
    gp->AddSample(VectorType::Ones(1)*4,VectorType::Ones(1));
    gp->Initialize();

    // compute gp kernel matrix (interval 0-5)
    unsigned number_of_samples = 50;
    VectorType mean = VectorType::Zero(number_of_samples);
    MatrixType K = MatrixType::Zero(number_of_samples,number_of_samples);
#pragma omp parallel for
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x1(1);
        x1(0) = i * 5.0/static_cast<double>(number_of_samples);
        mean[i] = gp->Predict(x1)(0);

        for(unsigned j=i; j<number_of_samples; j++){
            VectorType x2(1);
            x2(0) = j * 5.0/number_of_samples;

            K(i,j) = (*gp)(x1,x2);
            K(j,i) = K(i,j);
        }

    }


    // generate multivariate normal random numbers
    Eigen::SelfAdjointEigenSolver<MatrixType> eigenSolver(K);
    unsigned numComponentsToKeep = ((eigenSolver.eigenvalues().array() - 1e-10) > 0).count();
    MatrixType rot = eigenSolver.eigenvectors().rowwise().reverse().topLeftCorner(K.rows(), numComponentsToKeep);
    VectorType scl = eigenSolver.eigenvalues().reverse().topRows(numComponentsToKeep).array().sqrt();

    MatrixType Q = rot*scl.asDiagonal();
    double err = (Q*Q.transpose() - K).norm();
    if(err > 1e-8 || std::isnan(err)){
        std::cout << " [failed] eigen decomposition not accurate enough. (error: " << err << ")" << std::endl;
        return;
    }

    for(unsigned k=0; k<10; k++){
        VectorType r = rot * VectorType(GetRandomVector(number_of_samples).array() * scl.array()) + mean;

        if(std::fabs(r[10] - mean[10]) > 1e-9 ||
                std::fabs(r[20] - mean[20]) > 1e-9 ||
                std::fabs(r[30] - mean[30]) > 1e-9 ||
                std::fabs(r[40] - mean[40]) > 1e-9){
            std::cout << " [failed] samples do not corresponds to the landmarks." << std::endl;
            return;
        }
    }

    std::cout << " [passed]" << std::endl;
}


int main (int argc, char *argv[]){
    std::cout << "Gaussian process posterior test: " << std::endl;
    Test1();
    Test2();

    return 0;
}

