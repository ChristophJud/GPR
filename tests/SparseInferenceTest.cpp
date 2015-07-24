
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
#include "Prior.h"
#include "SparseLikelihood.h"

#include "GaussianProcessInference.h"



// setup sparse Gaussian process
typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;
typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef SparseGaussianProcessType::VectorType VectorType;
typedef SparseGaussianProcessType::MatrixType MatrixType;
typedef SparseGaussianProcessType::DiagMatrixType DiagMatrixType;
typedef SparseGaussianProcessType::VectorListType VectorListType;

typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;



// test the efficient inversion resp. determinant
void Test1(){
    std::cout << "Test 1.1 efficient inversion: ... " << std::flush;

    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.1;
    double jitter = 0.5;

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);


    // setup kernel
    double sigma = 0.23;
    double scale = 10;
    GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));

    // setup sparse gaussian process
    SparseGaussianProcessType::Pointer sgp(new SparseGaussianProcessType(gk));
    //sgp->DebugOn();
    sgp->SetSigma(noise);
    sgp->SetJitter(jitter);


    // large index set
    unsigned n = 1000;
    double start = -2;
    double stop = 5;
    VectorType Xn = VectorType::Zero(n);
    for(unsigned i=0; i<n; i++){
        Xn[i] = start + i*(stop-start)/n;
    }
    // fill up to sgp
    std::vector<double> dense_y;
    std::vector<double> sparse_y;
    for(unsigned i=0; i<n; i++){
        double v = f(Xn[i])+r();
        dense_y.push_back(v);
        sgp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,v));
    }

    // small index set
    unsigned m = 25;
    VectorType Xm = VectorType::Zero(m);
    for(unsigned i=0; i<m; i++){
        Xm[i] = start + i*(stop-start)/m;
    }
    // fill up to sgp
    for(unsigned i=0; i<m; i++){
        double v = f(Xm[i])+r();
        sparse_y.push_back(v);
        sgp->AddInducingSample(VectorType::Constant(1,Xm[i]), VectorType::Constant(1,v));
    }

    // construct Gaussian log likelihood
    typedef gpr::SparseGaussianLogLikelihood<double> SparseGaussianLogLikelihoodType;
    typedef typename SparseGaussianLogLikelihoodType::Pointer SparseGaussianLogLikelihoodTypePointer;
    SparseGaussianLogLikelihoodTypePointer sgl(new SparseGaussianLogLikelihoodType());

    // get all the important matrices
    MatrixType K;
    MatrixType K_inv;
    MatrixType Knm;
    DiagMatrixType I_sigma;

    sgl->GetCoreMatrices(sgp, K, K_inv, Knm, I_sigma);

    //---------------------------------------------------------------------------
    // inversion test
    MatrixType D;
    sgl->EfficientInversion(sgp, D, I_sigma, K_inv, K, Knm);

    MatrixType T = (MatrixType(I_sigma)+Knm*K_inv*Knm.adjoint()).inverse();
    double inv_error = (D-T).norm();

    if(inv_error < 1e-7){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::stringstream ss; ss<<inv_error; throw ss.str();
    }

    std::cout << "Test 1.2 efficient determinant: ... " << std::flush;
    //---------------------------------------------------------------------------
    // determinant test
    double determinant = sgl->EfficientDeterminant(I_sigma, K_inv, K, Knm);
    double det_err = std::fabs(determinant-(MatrixType(I_sigma)+Knm*K_inv*Knm.adjoint()).determinant());

    if(det_err < 1e-10){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::stringstream ss; ss<<det_err; throw ss.str();
    }

}

void Test2(double jitter){
    std::cout << "Test 2.1 core matrix test (jitter=" << jitter << "): ... " << std::flush;

    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.01;


    // setup kernel
    double sigma = 0.23;
    double scale = 10;
    GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));

    // setup sparse gaussian process
    SparseGaussianProcessType::Pointer sgp(new SparseGaussianProcessType(gk));
    //sgp->DebugOn();
    sgp->SetSigma(noise);
    sgp->SetJitter(jitter);


    // large index set
    unsigned n = 10;
    double start = -2;
    double stop = 5;
    VectorType Xn = VectorType::Zero(n);
    for(unsigned i=0; i<n; i++){
        Xn[i] = start + i*(stop-start)/n;
    }
    // fill up to sgp
    std::vector<double> dense_y;
    std::vector<double> sparse_y;
    for(unsigned i=0; i<n; i++){
        double v = f(Xn[i]);
        dense_y.push_back(v);
        sgp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,v));
    }

    // small index set
    unsigned m = 10;
    VectorType Xm = VectorType::Zero(m);
    for(unsigned i=0; i<m; i++){
        Xm[i] = start + i*(stop-start)/m;
    }
    // fill up to sgp
    for(unsigned i=0; i<m; i++){
        double v = f(Xm[i]);
        sparse_y.push_back(v);
        sgp->AddInducingSample(VectorType::Constant(1,Xm[i]), VectorType::Constant(1,v));
    }

    // construct Gaussian log likelihood
    typedef gpr::SparseGaussianLogLikelihood<double> SparseGaussianLogLikelihoodType;
    typedef typename SparseGaussianLogLikelihoodType::Pointer SparseGaussianLogLikelihoodTypePointer;
    SparseGaussianLogLikelihoodTypePointer sgl(new SparseGaussianLogLikelihoodType());

    // get all the important matrices
    MatrixType K;
    MatrixType K_inv;
    MatrixType Kmn;
    DiagMatrixType I_sigma;

    sgl->GetCoreMatrices(sgp, K, K_inv, Kmn, I_sigma);


    //---------------------------------------------------------------------------
    // core matrix test
    MatrixType C; //
    sgl->GetCoreMatrix(sgp, C, K_inv, Kmn);

    MatrixType M;
    sgp->ComputeDenseKernelMatrix(M);

    double err = (C-M).norm();

    if((err < 1e-2 && jitter > 0) || (err < 2000 && jitter == 0)){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::stringstream ss; ss<<err; throw ss.str();
    }


    std::cout << "Test 2.2 core matrix trace test (jitter=" << jitter << "): ... " << std::flush;
    if(M.trace() == sgl->GetKernelMatrixTrace(sgp)){
        std::cout << "[passed]" << std::endl;
    }
    else{
        throw std::string("error in calculating kernel matrix trace.");
    }
}

void Test3(){
    std::cout << "Test 3: sparse likelihood gradient test... " << std::flush;

    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.2;
    double jitter = 0.01;

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);


    double h = 0.0001;
    bool passed = true;

    for(unsigned iter=0; iter<10; iter++){

        // setup kernel
        double scale = 10;
        double sigma = 0.1 + iter*0.02;


        GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));

        // setup sparse gaussian process
        SparseGaussianProcessType::Pointer sgp(new SparseGaussianProcessType(gk));
        //sgp->DebugOn();
        sgp->SetSigma(noise);
        sgp->SetJitter(jitter);

        // large index set
        //unsigned n = 100;
        unsigned n = 50 + iter;
        double start = -2;
        double stop = 5;
        VectorType Xn = VectorType::Zero(n);
        for(unsigned i=0; i<n; i++){
            Xn[i] = start + i*(stop-start)/n;
        }
        // fill up to sgp
        std::vector<double> dense_y;
        std::vector<double> sparse_y;
        for(unsigned i=0; i<n; i++){
            double v = f(Xn[i])+r();
            dense_y.push_back(v);
            sgp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,v));
        }

        // small index set
        //unsigned m = 20;
        unsigned m = 5 + iter;
        VectorType Xm = VectorType::Zero(m);
        for(unsigned i=0; i<m; i++){
            Xm[i] = start + i*(stop-start)/m;
        }
        // fill up to sgp
        for(unsigned i=0; i<m; i++){
            double v = f(Xm[i])+r();
            sparse_y.push_back(v);
            sgp->AddInducingSample(VectorType::Constant(1,Xm[i]), VectorType::Constant(1,v));
        }


        // construct Gaussian log likelihood
        typedef gpr::SparseGaussianLogLikelihood<double> SparseGaussianLogLikelihoodType;
        typedef typename SparseGaussianLogLikelihoodType::Pointer SparseGaussianLogLikelihoodTypePointer;
        SparseGaussianLogLikelihoodTypePointer sgl(new SparseGaussianLogLikelihoodType());

        //sgl->DebugOn();


        VectorType D = sgl->GetParameterDerivatives(sgp);

        {
            // central differences sigma
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma+h/2, scale)));
            double plus = (*sgl)(sgp)[0];
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma-h/2, scale)));
            double minus = (*sgl)(sgp)[0];

            double d = (plus - minus)/h;

            if(std::fabs(D[0] - d) > 1){
                passed = false;
                std::cout <<  "central difference (sigma): " << d << ", but is " << D[0] <<   std::endl;
            }
        }
        {
            // central differences scale
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma, scale+h/2)));
            double plus = (*sgl)(sgp)[0];
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma, scale-h/2)));
            double minus = (*sgl)(sgp)[0];

            double d = (plus - minus)/h;

            if(std::fabs(D[1] - d) > 0.1){
                passed = false;
                std::cout <<  "central difference (scale): " << d << ", but is " << D[1] <<   std::endl;
            }
        }
    }

    if(passed){
        std::cout << "[passed]" << std::endl;
    }
    else{
        throw std::string("errors in central differences calculation");
    }

    return;
}


void Test4(){
    //std::cout << "Test 4: sparse maximum gaussian log likelihood with gradient descent test ..." << std::flush;
    std::cout.precision(8);

    typedef gpr::GaussianExpKernel<double>           GaussianExpKernelType;
    typedef GaussianExpKernelType::Pointer           GaussianExpKernelTypePointer;
    typedef gpr::GaussianKernel<double>              GaussianKernelType;
    typedef GaussianKernelType::Pointer              GaussianKernelTypePointer;
    typedef gpr::SparseGaussianLogLikelihood<double>       LikelihoodType;
    typedef LikelihoodType::Pointer                  LikelihoodTypePointer;
    typedef gpr::GaussianProcessInference<double>    GaussianProcessInferenceType;
    typedef GaussianProcessInferenceType::Pointer    GaussianProcessInferenceTypePointer;

    typedef gpr::GaussianProcess<double> GaussianProcessType;
    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;
    typedef GaussianProcessType::DiagMatrixType DiagMatrixType;
    typedef GaussianProcessType::VectorListType VectorListType;

    // global parameters
    unsigned n = 300;
    unsigned m = 50;
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
    SparseGaussianProcessType::Pointer gp(new SparseGaussianProcessType(gk, 0.11));
    //gp->DebugOn();
    gp->SetSigma(noise);
    for(unsigned i=0; i<n; i++){
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,Yn[i]));
    }

    std::vector<unsigned> indices;
    for(unsigned i=0; i<n; i++){
        indices.push_back(i);
    }
    std::random_shuffle(indices.begin(), indices.end());

    for(unsigned i=0; i<m; i++){
        gp->AddInducingSample(VectorType::Constant(1,Xn[indices[i]]), VectorType::Constant(1,Yn[indices[i]]));
    }

    // setup likelihood
    double step = 1e-1;
    unsigned iterations = 100;

    LikelihoodTypePointer lh(new LikelihoodType());
    GaussianProcessInferenceTypePointer gpi(new GaussianProcessInferenceType(lh, gp, step, iterations));

    bool exp_output = true;
    gpi->Optimize(true, exp_output);


    std::cout << "print \"Parameters are: ";
    GaussianProcessInferenceType::ParameterVectorType parameters = gpi->GetParameters();
    for(unsigned i=0; i<parameters.size(); i++){
        parameters[i] = std::exp(parameters[i]);
        std::cout << parameters[i] << ", ";
    }
    std::cout << "\"" << std::endl;


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


//    if(error/gt_n > 2){
//        std::cout << "[failed] with an avg error of " << error/gt_n << std::endl;
//    }
//    else{
//        std::cout << "[passed]" << std::endl;
//    }
return;

    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;

    std::cout << "x = np.array([" << std::endl;
    for(unsigned i=0; i<gt_n; i++){
        std::cout << start + i*(stop-start)/gt_n << ", ";
    }
    std::cout << "])" << std::endl;

    std::cout << "y = np.array([" << std::endl;
    for(unsigned i=0; i<gt_n; i++){
        std::cout << f(start + i*(stop-start)/gt_n) << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x,y)" << std::endl;

    std::cout << "xp = np.array([" << std::endl;
    for(unsigned i=0; i<m; i++){
        std::cout << Xn[indices[i]] << ", ";
    }
    std::cout << "])" << std::endl;

    std::cout << "yp = np.array([" << std::endl;
    for(unsigned i=0; i<m; i++){
        std::cout << Yn[indices[i]] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(xp,yp, '.k')" << std::endl;

    std::cout << "Y = np.array([" << std::endl;
    for(unsigned i=0; i<gt_n; i++){
        std::cout << gp->Predict(VectorType::Constant(1,(start + i*(stop-start)/gt_n)))[0] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x,Y, '-r')" << std::endl;


    std::cout << "plt.show()" << std::endl;
}

int main (int argc, char *argv[]){
    //std::cout << "Sparse Gaussian Process test: " << std::endl;
    try{
//        Test1();
//        Test2(0); // jitter 0
//        Test2(0.001); // jitter 0.001
//        Test3();
        Test4();
    }
    catch(std::string& s){
        std::cout << "[failed] Error: " << s << std::endl;
        return -1;
    }

    return 0;
}
