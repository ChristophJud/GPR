
#include <string>
#include <iostream>
#include <memory>

#include <boost/random.hpp>

#include <Eigen/Dense>

#include "GaussianProcess.h"
#include "SparseGaussianProcess.h"

#include "Likelihood.h"
#include "SparseLikelihood.h"


void Test1(){

    // setup sparse Gaussian process
    typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;
    typedef gpr::GaussianProcess<double> GaussianProcessType;
    typedef SparseGaussianProcessType::VectorType VectorType;
    typedef SparseGaussianProcessType::MatrixType MatrixType;


    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.1;
    double jitter = 0.01;

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

    // setup dense gaussian process
    GaussianProcessType::Pointer gp(new GaussianProcessType(gk));
    //gp->DebugOn();
    gp->SetSigma(noise);

    // setup dense gaussian process with sparse samples
    GaussianProcessType::Pointer gp_test(new GaussianProcessType(gk));
    //gp->DebugOn();
    gp_test->SetSigma(noise);


    // large index set
    unsigned n = 100;
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
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,v));
    }

    // small index set
    unsigned m = 20;
    VectorType Xm = VectorType::Zero(m);
    for(unsigned i=0; i<m; i++){
        Xm[i] = start + i*(stop-start)/m;
    }
    // fill up to sgp
    for(unsigned i=0; i<m; i++){
        double v = f(Xm[i])+r();
        sparse_y.push_back(v);
        sgp->AddInducingSample(VectorType::Constant(1,Xm[i]), VectorType::Constant(1,v));
        gp_test->AddSample(VectorType::Constant(1,Xm[i]), VectorType::Constant(1,v));
    }

    //std::cout << "Initializing sparse GP..." << std::flush;
    sgp->Initialize();
    //std::cout << std::endl << "Initializing dense GP... " << std::flush;
    gp->Initialize();
    gp_test->Initialize();
    //std::cout << std::endl;


    // construct Gaussian log likelihood
    if(false){
        typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
        typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
        GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());
        std::cout << "Likelihood before optimization: " << (*gl)(gp) << std::endl;

        double lambda = 1e-5;
        for(unsigned i=0; i<20; i++){
            // analytical
            try{
                GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
                gp->SetKernel(gk);

                //std::cout << "Likelihood " << (*gl)(gp) << ", sigma/scale " << sigma << "/" << scale << std::endl;

                VectorType likelihood_update = gl->GetParameterDerivatives(gp);

                sigma += lambda * likelihood_update[0];
                scale += lambda * likelihood_update[1];
            }
            catch(std::string& s){
                std::cout << "[failed] " << s << std::endl;
                return;
            }
        }

        std::cout << "Likelihood after optimization: " <<  (*gl)(gp) << std::endl;
        std::cout << "New sigma/scale: " << sigma << "/" << scale << std::endl;
    }

    // construct Gaussian log likelihood
    typedef gpr::SparseGaussianLogLikelihood<double> SparseGaussianLogLikelihoodType;
    typedef typename SparseGaussianLogLikelihoodType::Pointer SparseGaussianLogLikelihoodTypePointer;
    SparseGaussianLogLikelihoodTypePointer sgl(new SparseGaussianLogLikelihoodType());


    std::cout <<  "print " << (*sgl)(sgp) << std::endl;

    // todo: test efficient inversion / determinant

//    double sigma = 0.1;
//    double scale = 1;
//    GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
//    sgl->SetKernel(new GaussianKernelType(1, scale));

    return;

    std::cout << "import numpy as np" << std::endl;
    std::cout << "import pylab as plt" << std::endl;

    // ground truth
    unsigned gt_n = 1000;
    std::cout << "x = np.array([";
    for(unsigned i=0; i<gt_n; i++){
        std::cout << start + i*(stop-start)/gt_n << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "y = np.array([";
    for(unsigned i=0; i<gt_n; i++){
        std::cout << f(start + i*(stop-start)/gt_n) << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x,y)" << std::endl;

    // dense samples
    std::cout << "dense_x = np.array([";
    for(unsigned i=0; i<n; i++){
        std::cout << Xn[i] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "dense_y = np.array([";
    for(unsigned i=0; i<n; i++){
        std::cout << dense_y[i] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(dense_x, dense_y, '*k')" << std::endl;


    // dense prediction
    double dense_error = 0;
    std::cout << "gp_y = np.array([";
    for(unsigned i=0; i<gt_n; i++){
        double x = start + i*(stop-start)/gt_n;
        double p = gp->Predict(VectorType::Constant(1,x))[0];
        dense_error += std::fabs(p-f(x));
        std::cout << p << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x, gp_y, '-r')" << std::endl;


    // sparse samples
    std::cout << "sparse_x = np.array([";
    for(unsigned i=0; i<m; i++){
        std::cout << Xm[i] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "sparse_y = np.array([";
    for(unsigned i=0; i<m; i++){
        std::cout << sparse_y[i] << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(sparse_x, sparse_y, 'ok')" << std::endl;

    // sparse prediction
    double sparse_error = 0;
    std::cout << "sgp_y = np.array([";
    for(unsigned i=0; i<gt_n; i++){
        double x = start + i*(stop-start)/gt_n;
        double p = sgp->Predict(VectorType::Constant(1,x))[0];
        sparse_error += std::fabs(p-f(x));
        std::cout << p << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x, sgp_y, '-g')" << std::endl;

    // test prediction
    double test_error = 0;
    std::cout << "gp_test_y = np.array([";
    for(unsigned i=0; i<gt_n; i++){
        double x = start + i*(stop-start)/gt_n;
        double p = gp_test->Predict(VectorType::Constant(1,x))[0];
        test_error += std::fabs(p-f(x));
        std::cout << p << ", ";
    }
    std::cout << "])" << std::endl;
    std::cout << "plt.plot(x, gp_test_y, '-k')" << std::endl;

    std::cout << "plt.show()" << std::endl;

    std::cout << "t=\"\"\"" << std::endl;
    std::cout << "---- Sparse----" << std::endl;
    std::cout << "Prediction: " << std::endl;
    std::cout << sgp->Predict(VectorType::Zero(1)) << std::endl;
    std::cout << "Variance: " << std::endl;
    std::cout << (*sgp)(VectorType::Zero(1), VectorType::Constant(1,0.2)) << std::endl;
    std::cout << "Error: " << std::endl;
    std::cout << sparse_error << std::endl;


    std::cout << "---- Dense----" << std::endl;
    std::cout << "Prediction: " << std::endl;
    std::cout << gp->Predict(VectorType::Zero(1)) << std::endl;
    std::cout << "Variance: " << std::endl;
    std::cout << (*gp)(VectorType::Zero(1), VectorType::Constant(1,0.2)) << std::endl;
    std::cout << "Error: " << std::endl;
    std::cout << dense_error << std::endl;

    std::cout << "---- Test----" << std::endl;
    std::cout << "Prediction: " << std::endl;
    std::cout << gp_test->Predict(VectorType::Zero(1)) << std::endl;
    std::cout << "Variance: " << std::endl;
    std::cout << (*gp_test)(VectorType::Zero(1), VectorType::Constant(1,0.2)) << std::endl;
    std::cout << "Error: " << std::endl;
    std::cout << test_error << std::endl;

    std::cout << "\"\"\"" << std::endl;
    std::cout << "print(t)" << std::endl;

}


// test the efficient inversion resp. determinant
void Test2(){
    std::cout << "Test 2.1 efficient inversion: ... " << std::flush;

    // setup sparse Gaussian process
    typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;
    typedef SparseGaussianProcessType::VectorType VectorType;
    typedef SparseGaussianProcessType::MatrixType MatrixType;
    typedef SparseGaussianProcessType::DiagMatrixType DiagMatrixType;


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

    //std::cout << "Initializing sparse GP..." << std::flush;
    sgp->Initialize();


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
    // inversion test
    MatrixType D;
    sgl->EfficientInversion(sgp, D, I_sigma, K_inv, K, Kmn);

    MatrixType T = (MatrixType(I_sigma)+Kmn*K_inv*Kmn.adjoint()).inverse();
    double inv_error = (D-T).norm();

    if(inv_error < 1e-8){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }

    std::cout << "Test 2.2 efficient determinant: ... " << std::flush;
    //---------------------------------------------------------------------------
    // determinant test
    double determinant = sgl->EfficientDeterminant(I_sigma, K_inv, K, Kmn);
    double det_err = std::fabs(determinant-(MatrixType(I_sigma)+Kmn*K_inv*Kmn.adjoint()).determinant());

    if(det_err < 1e-10){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }

}

void Test3(double jitter){
    std::cout << "Test 3.1 core matrix test (jitter=" << jitter << "): ... " << std::flush;

    // setup sparse Gaussian process
    typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;
    typedef SparseGaussianProcessType::VectorType VectorType;
    typedef SparseGaussianProcessType::MatrixType MatrixType;
    typedef SparseGaussianProcessType::DiagMatrixType DiagMatrixType;

    typedef gpr::GaussianProcess<double> GaussianProcessType;

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

    //std::cout << "Initializing sparse GP..." << std::flush;
    sgp->Initialize();


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
        std::cout << "[failed] with an error of " << err << std::endl;
    }


    std::cout << "Test 3.2 core matrix trace test (jitter=" << jitter << "): ... " << std::flush;
    if(M.trace() == sgl->GetKernelMatrixTrace(sgp)){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }
}

int main (int argc, char *argv[]){
    //std::cout << "Sparse Gaussian Process test: " << std::endl;
    try{
        Test1();
        //Test2();
        //Test3(0); // jitter 0
        //Test3(0.001); // jitter 0.001
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
