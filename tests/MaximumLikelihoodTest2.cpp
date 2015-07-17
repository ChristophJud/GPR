

#include <string>
#include <iostream>
#include <memory>
#include <vector>

#include <boost/random.hpp>

#include <Eigen/Dense>

#include "Kernel.h"
#include "GaussianProcess.h"
#include "SparseGaussianProcess.h"

#include "Likelihood.h"
#include "SparseLikelihood.h"

#include "GaussianProcessInference.h"


typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;
typedef GaussianProcessType::DiagMatrixType DiagMatrixType;
typedef GaussianProcessType::VectorListType VectorListType;

typedef gpr::GaussianProcessInference<double>    GaussianProcessInferenceType;
typedef GaussianProcessInferenceType::Pointer    GaussianProcessInferenceTypePointer;
typedef gpr::GaussianLogLikelihood<double>       LikelihoodType;
typedef LikelihoodType::Pointer                  LikelihoodTypePointer;
typedef gpr::GaussianExpKernel<double>           GaussianExpKernelType;
typedef GaussianExpKernelType::Pointer           GaussianExpKernelTypePointer;
typedef gpr::GaussianKernel<double>              GaussianKernelType;
typedef GaussianKernelType::Pointer              GaussianKernelTypePointer;

void Test1(){
    std::cout << "Test 1: maximum gaussian log likelihood with gradient descent test ..." << std::flush;
    std::cout.precision(8);

    // global parameters
    unsigned n = 200;
    double noise = 0.1;

    // construct training data
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double start = -5;
    double stop = 10;
    VectorType Xn = VectorType::Zero(n);
    VectorType Yn = VectorType::Zero(n);
    for(unsigned i=0; i<n; i++){
        Xn[i] = start + i*(stop-start)/n;
        Yn[i] = f(Xn[i])+r();
    }


    GaussianExpKernelTypePointer gk(new GaussianExpKernelType(1, 1));
    GaussianProcessType::Pointer gp(new GaussianProcessType(gk));
    //gp->DebugOn();
    gp->SetSigma(noise);
    for(unsigned i=0; i<n; i++){
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,Yn[i]));
    }



    // setup likelihood
    double step = 1e-1;
    unsigned iterations = 100;

    LikelihoodTypePointer lh(new LikelihoodType());
    GaussianProcessInferenceTypePointer gpi(new GaussianProcessInferenceType(lh, gp, step, iterations));

    bool exp_output = true;
    gpi->Optimize(false, exp_output);


    //std::cout << "Parameters are: ";
    GaussianProcessInferenceType::ParameterVectorType parameters = gpi->GetParameters();
    for(unsigned i=0; i<parameters.size(); i++){
        parameters[i] = std::exp(parameters[i]);
        //std::cout << parameters[i] << ", ";
    }
    //std::cout << std::endl;


    GaussianKernelTypePointer k(new GaussianKernelType(1,1));
    k->SetParameters(parameters);
    gp->SetKernel(k);


    // evaluate error
    double error = 0;
    unsigned gt_n = 1000;
    for(unsigned i=0; i<gt_n; i++){
        double x = start + i*(stop-start)/gt_n;
        double p = gp->Predict(VectorType::Constant(1,x))[0];
        error += std::fabs(p-f(x));
    }


    if(error/gt_n > 2){
        std::cout << "[failed] with an avg error of " << error/gt_n << std::endl;
    }
    else{
        std::cout << "[passed]" << std::endl;
    }

}


typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;

void Test2(){
    std::cout << "Test 2: maximum gaussian log likelihood on dense gp and prediction with sparse gp test ... " << std::flush;
    std::cout.precision(8);

    // global parameters
    unsigned n = 300;
    unsigned m = 50;
    double noise = 0.1;
    double jitter = 0.1;

    // construct training data
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);

    double start = -5;
    double stop = 10;
    VectorType Xn = VectorType::Zero(n);
    VectorType Yn = VectorType::Zero(n);
    for(unsigned i=0; i<n; i++){
        Xn[i] = start + i*(stop-start)/n;
        Yn[i] = f(Xn[i])+r();
    }

    std::vector<unsigned> indices;
    for(unsigned i=0; i<n; i++){
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end());


    // setup dense gaussian process
    GaussianExpKernelTypePointer gk(new GaussianExpKernelType(1, 1));
    GaussianProcessType::Pointer gp(new GaussianProcessType(gk));
    gp->SetSigma(noise);

    // setup sparse gaussian process
    SparseGaussianProcessType::Pointer sgp(new SparseGaussianProcessType(gk));
    sgp->SetSigma(noise);
    sgp->SetJitter(jitter);

    for(unsigned i=0; i<m; i++){
        gp->AddSample(VectorType::Constant(1,Xn[indices[i]]), VectorType::Constant(1,Yn[indices[i]]));
        sgp->AddInducingSample(VectorType::Constant(1,Xn[indices[i]]), VectorType::Constant(1,Yn[indices[i]]));
    }
    for(unsigned i=0; i<n; i++){
        sgp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,Yn[i]));
    }



    // setup likelihood
    double step = 1e-1;
    unsigned iterations = 100;

    LikelihoodTypePointer lh(new LikelihoodType());
    GaussianProcessInferenceTypePointer gpi(new GaussianProcessInferenceType(lh, gp, step, iterations));

    bool exp_output = true;
    gpi->Optimize(false, exp_output);



    //std::cout << "Parameters are: ";
    GaussianProcessInferenceType::ParameterVectorType parameters = gpi->GetParameters();
    for(unsigned i=0; i<parameters.size(); i++){
        parameters[i] = std::exp(parameters[i]);
        //std::cout << parameters[i] << ", ";
    }
    //std::cout << std::endl;


    GaussianKernelTypePointer k(new GaussianKernelType(1,1));
    k->SetParameters(parameters);
    gp->SetKernel(k);
    sgp->SetKernel(k);



    // evaluate error
    double dense_error = 0;
    double sparse_error = 0;
    unsigned gt_n = 1000;
    for(unsigned i=0; i<gt_n; i++){
        double x = start + i*(stop-start)/gt_n;
        double pd = gp->Predict(VectorType::Constant(1,x))[0];
        double ps = sgp->Predict(VectorType::Constant(1,x))[0];
        dense_error += std::fabs(pd-f(x));
        sparse_error += std::fabs(ps-f(x));
    }


    if(dense_error < sparse_error){
        std::cout << "[failed]" << std::endl;
    }
    else{
        std::cout << "[passed]" << std::endl;
    }


}

int main (int argc, char *argv[]){
    std::cout << "Maximum likelihood test 2: " << std::endl;
    try{
        Test1();
        Test2();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
