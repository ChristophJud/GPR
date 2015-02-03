// compilation: g++ -I /home/jud/apps/Eigen_3.1.3/linux64/include/eigen3/ gp_main.cpp -o gpTest -std=c++0x

#include <iostream>
#include <memory>
#include <ctime>

#include <boost/random.hpp>

#include <Eigen/Cholesky>

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

    std::cout << std::endl << "x = np.array([";
    // add training samples
    for(unsigned i=0; i<number_of_samples; i++){
        VectorType x(1);
        x(0) = i * 2*M_PI/number_of_samples;

        VectorType y(1);
        y(0) = std::sin(x(0));
        gp->AddSample(x,y);

        std::cout << x(0) << ", ";
    }
    std::cout << "])" << std::endl;
    gp->Initialize();

    {
        // calculate credible interval
        std::cout << "p = np.array([";

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
            std::cout << gp->Predict(x)(0) << ", ";
        }
        std::cout << "])" << std::endl;

    }

    {
        // compute gp kernel matrix
        MatrixType K = MatrixType::Zero(number_of_samples,number_of_samples);
#pragma omp parallel for
        for(unsigned i=0; i<number_of_samples; i++){
            VectorType x(1);
            x(0) = i * 2*M_PI/number_of_samples;
            for(unsigned j=i; j<number_of_samples; j++){
                VectorType y(1);
                y(0) = j * 2*M_PI/number_of_samples;

                K(i,j) = (*gp)(x,y);
                K(j,i) = K(i,j);
            }
        }

        // correlation matrix: diag(K)^(-0.5) K diag(K)^(-0.5)
        VectorType K_diag_sqrt = 1 / K.diagonal().array().sqrt();
        MatrixType C = K_diag_sqrt.asDiagonal() * K * K_diag_sqrt.asDiagonal();

        // perform cholesky decomposition of K
        Eigen::LLT<MatrixType> llt;
        llt.compute(C);


        if((MatrixType(llt.matrixU()).transpose() * MatrixType(llt.matrixU()) - C).norm() > 1e-12){
            std::cout << " [failed] cholesky decomposition not accurate enough." << std::endl;
            //return;
        }

        std::cout << std::endl;
        std::cout << "r = []" << std::endl;
        for(unsigned k=0; k<10; k++){
            std::cout << "r.append(np.array([";
            VectorType r = GetRandomVector(number_of_samples).transpose() * MatrixType(llt.matrixU());

            for(unsigned i=0; i<r.rows(); i++){
                std::cout << r[i] << ", ";
            }
            std::cout << "]))" << std::endl;
        }

    }

    std::cout << " [passed]" << std::endl;
}


int main (int argc, char *argv[]){
    std::cout << "Gaussian process posterior test: " << std::endl;
    Test1();

    return 0;
}

