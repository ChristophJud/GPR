

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

/**
 * An SVD based implementation of the Moore-Penrose pseudo-inverse
 */
template<class TMatrixType>
TMatrixType pinv(const TMatrixType& m, double epsilon = std::numeric_limits<double>::epsilon()) {
    typedef Eigen::JacobiSVD<TMatrixType> SVD;
    SVD svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    typedef typename SVD::SingularValuesType SingularValuesType;
    const SingularValuesType singVals = svd.singularValues();
    SingularValuesType invSingVals = singVals;
    for(int i=0; i<singVals.rows(); i++) {
        if(singVals(i) <= epsilon) {
            invSingVals(i) = 0.0; // FIXED can not be safely inverted
        }
        else {
            invSingVals(i) = 1.0 / invSingVals(i);
        }
    }
    return TMatrixType(svd.matrixV() *
            invSingVals.asDiagonal() *
            svd.matrixU().transpose());
}

template<class TScalarType>
class GaussianProcessInference{
public:
    typedef GaussianProcessInference Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef typename gpr::GaussianProcess<TScalarType>       GaussianProcessType;
    typedef typename GaussianProcessType::Pointer            GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorListType     VectorListType;
    typedef typename GaussianProcessType::VectorType         VectorType;
    typedef typename GaussianProcessType::MatrixType         MatrixType;
    typedef std::vector<TScalarType>    ParameterVectorType;

    typedef typename gpr::GaussianLogLikelihood<TScalarType> GaussianLogLikelihoodType;
    typedef typename GaussianLogLikelihoodType::Pointer      GaussianLogLikelihoodTypePointer;
    typedef typename GaussianLogLikelihoodType::ValueDerivativePair ValueDerivativePair;


    typedef gpr::GaussianKernel<double, gpr::KernelParameterType::Exponential>             GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

    GaussianProcessInference(GaussianProcessTypePointer gp, double stepwidth, unsigned iterations) :
        m_StepWidth(stepwidth), m_StepWidth3(stepwidth*stepwidth*stepwidth), m_NumberOfIterations(iterations){
        m_GaussianProcess = gp;

        m_Parameters.resize(gp->GetKernel()->GetNumberOfParameters(), 1);
    }

    ~GaussianProcessInference(){}

    ParameterVectorType GetParameters(){
        return m_Parameters;
    }

    void Optimize(bool output=true){
        GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());

        for(unsigned i=0; i<m_NumberOfIterations; i++){
            // analytical
            try{
                typedef gpr::GaussianKernel<double, gpr::KernelParameterType::Exponential>   GaussianKernelType;
                typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

                GaussianKernelTypePointer gk(new GaussianKernelType(m_Parameters[0], m_Parameters[1]));
                m_GaussianProcess->SetKernel(gk);
                //gp->DebugOn();

                ValueDerivativePair value_derivative = gl->GetValueAndParameterDerivatives(m_GaussianProcess);
                VectorType likelihood_gradient = value_derivative.second;
                VectorType likelihood = value_derivative.first;

                VectorType parameter_update = (pinv<MatrixType>(likelihood_gradient * likelihood_gradient.adjoint()) * likelihood_gradient);
                if(output){
                    std::cout << "Likelihood " << value_derivative.first << ", sigma/scale " << std::exp(m_Parameters[0]) << "/" << std::exp(m_Parameters[1]) << std::flush;
                    std::cout << ", Gradients: " << likelihood_gradient.adjoint() << std::flush;
                    std::cout << ", inf(J'J)J': " << parameter_update.adjoint() << std::flush;
                    std::cout << ", update: " << std::flush;
                }

                for(unsigned p=0; p<m_Parameters.size(); p++){
                    double u;
                    if(parameter_update[p]==0){ // log gradient step
                        if(likelihood_gradient[p]>=0){
                            u = m_StepWidth3*std::log(1+likelihood_gradient[p]);
                        }
                        else{
                            u = -m_StepWidth3*std::log(1+std::fabs(likelihood_gradient[p]));
                        }
                        m_Parameters[p] += u;
                    }
                    else{ // Gauss Newton step
                        u = parameter_update[p]*likelihood[0];
                        if(u>0){
                             u = m_StepWidth*std::log(1+u);
                        }
                        else{
                            u = -m_StepWidth*std::log(1+std::fabs(u));
                        }
                        m_Parameters[p] -= u;
                    }
                    if(output)std::cout << u << " " << std::flush;
                }

                if(output){
                    std::cout << ", new parameters: " << std::flush;
                    for(unsigned p=0; p<m_Parameters.size(); p++){
                        std::cout << std::exp(m_Parameters[p]) << ", " << std::flush;
                    }
                    std::cout << std::endl;
                }
            }
            catch(std::string& s){
                std::cout << "[failed] " << s << std::endl;
                return;
            }
        }
    }

private:
    TScalarType m_StepWidth;
    TScalarType m_StepWidth3;
    unsigned m_NumberOfIterations;

    GaussianProcessTypePointer m_GaussianProcess;
    ParameterVectorType m_Parameters;

    GaussianProcessInference(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

// setup sparse Gaussian process
typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef GaussianProcessType::VectorType VectorType;
typedef GaussianProcessType::MatrixType MatrixType;
typedef GaussianProcessType::DiagMatrixType DiagMatrixType;
typedef GaussianProcessType::VectorListType VectorListType;

typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

void Test1(){
    std::cout.precision(8);

    // global parameters
    unsigned n = 100;
    double noise = 0.02;

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



    // setup gaussian process
    typedef gpr::GaussianProcess<double> GaussianProcessType;
    typedef gpr::GaussianProcess<double> GaussianProcessType;
    typedef GaussianProcessType::VectorType VectorType;
    typedef GaussianProcessType::MatrixType MatrixType;
    typedef GaussianProcessType::DiagMatrixType DiagMatrixType;
    typedef GaussianProcessType::VectorListType VectorListType;

    typedef gpr::GaussianKernel<double>             GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

    GaussianKernelTypePointer gk(new GaussianKernelType(1, 1));
    GaussianProcessType::Pointer gp(new GaussianProcessType(gk));
    //gp->DebugOn();
    gp->SetSigma(noise);
    for(unsigned i=0; i<n; i++){
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,Yn[i]));
    }


    // setup likelihood
    double step = 1e-1;
    unsigned iterations = 200;

    typedef GaussianProcessInference<double>    GaussianProcessInferenceType;
    typedef typename GaussianProcessInferenceType::Pointer    GaussianProcessInferenceTypePointer;
    GaussianProcessInferenceTypePointer gpi(new GaussianProcessInferenceType(gp, step, iterations));

    gpi->Optimize(false);

    GaussianProcessInferenceType::ParameterVectorType parameters = gpi->GetParameters();
    gp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(parameters[0], parameters[1])));

    //gp->Initialize();





    //return;

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

//    // dense samples
//    std::cout << "dense_x = np.array([";
//    for(unsigned i=0; i<n; i++){
//        std::cout << Xn[i] << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "dense_y = np.array([";
//    for(unsigned i=0; i<n; i++){
//        std::cout << dense_y[i] << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "plt.plot(dense_x, dense_y, '*k')" << std::endl;


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


//    // sparse samples
//    std::cout << "sparse_x = np.array([";
//    for(unsigned i=0; i<m; i++){
//        std::cout << Xm[i] << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "sparse_y = np.array([";
//    for(unsigned i=0; i<m; i++){
//        std::cout << sparse_y[i] << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "plt.plot(sparse_x, sparse_y, 'ok')" << std::endl;

//    // sparse prediction
//    double sparse_error = 0;
//    std::cout << "sgp_y = np.array([";
//    for(unsigned i=0; i<gt_n; i++){
//        double x = start + i*(stop-start)/gt_n;
//        double p = sgp->Predict(VectorType::Constant(1,x))[0];
//        sparse_error += std::fabs(p-f(x));
//        std::cout << p << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "plt.plot(x, sgp_y, '-g')" << std::endl;

//    // test prediction
//    double test_error = 0;
//    std::cout << "gp_test_y = np.array([";
//    for(unsigned i=0; i<gt_n; i++){
//        double x = start + i*(stop-start)/gt_n;
//        double p = gp_test->Predict(VectorType::Constant(1,x))[0];
//        test_error += std::fabs(p-f(x));
//        std::cout << p << ", ";
//    }
//    std::cout << "])" << std::endl;
//    std::cout << "plt.plot(x, gp_test_y, '-k')" << std::endl;

    std::cout << "plt.show()" << std::endl;

}

int main (int argc, char *argv[]){
    //std::cout << "Gaussian Process test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
