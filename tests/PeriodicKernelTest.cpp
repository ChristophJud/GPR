
// compilation: g++ -I /home/jud/apps/Eigen_3.1.3/linux64/include/eigen3/ gp_main.cpp -o gpTest -std=c++0x

#include <iostream>
#include <memory>
#include <ctime>
#include <cmath>

#include <boost/random.hpp>

#include "GaussianProcess.h"
#include "Kernel.h"
#include "MatrixIO.h"

using namespace gpr;

typedef PeriodicKernel<double>		DPKernelType;
typedef std::shared_ptr<DPKernelType> DPKernelTypePointer;
typedef GaussianProcess<double> DPGaussianProcessType;
typedef std::shared_ptr<DPGaussianProcessType> DPGaussianProcessTypePointer;

typedef PeriodicKernel<float>		KernelType;
typedef std::shared_ptr<KernelType> KernelTypePointer;
typedef GaussianProcess<float> GaussianProcessType;
typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

typedef DPGaussianProcessType::VectorType DPVectorType;
typedef DPGaussianProcessType::MatrixType DPMatrixType;
typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;


void Test1(){
    /*
     * Test 1: regression of a periodic signal (1D)
     * - generate a periodic signal, and try to predict some more periods
     * - ground truth is: sin(x)*cos(2.2*sin(x))
     */
    std::cout << "Test 1: periodic signal regression ..." << std::flush;

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

    KernelTypePointer k(new KernelType(0.77, 0.5, 0.4)); // scale, period, smoothness
    GaussianProcessTypePointer gp(new GaussianProcessType(k));
    gp->SetSigma(noise); // noise

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

    double err = (y-y_predict).norm();
    if(err>0.4){
        std::cout << " [failed]." << std::endl;
    }
    else{
        std::cout << " [passed]." << std::endl;
    }

}

void Test2(){
    /*
     * Test 2: gaussian process save and load
     */
    std::cout << "Test 2: save/load gaussian process... " << std::flush;
    KernelTypePointer k(new KernelType(1.8, 2.2, 0.3));
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


    KernelTypePointer k_dummy(new KernelType(1, 1, 1));
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
    std::cout << "Periodic kernel test: " << std::endl;
    Test1();
    Test2();

    return 0;
}

