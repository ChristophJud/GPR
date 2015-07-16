
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
class SparseGaussianProcessInference{
public:
    typedef SparseGaussianProcessInference Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef typename gpr::SparseGaussianProcess<TScalarType>    SparseGaussianProcessType;
    typedef typename SparseGaussianProcessType::Pointer         SparseGaussianProcessTypePointer;
    typedef typename SparseGaussianProcessType::VectorListType  VectorListType;
    typedef typename SparseGaussianProcessType::VectorType      VectorType;
    typedef typename SparseGaussianProcessType::MatrixType      MatrixType;
    typedef typename SparseGaussianProcessType::KernelType         KernelType;
    typedef typename SparseGaussianProcessType::KernelTypePointer  KernelTypePointer;
    typedef typename KernelType::ParameterType                  ParameterType;
    typedef typename KernelType::ParameterVectorType            ParameterVectorType;

    typedef typename gpr::SparseGaussianLogLikelihood<TScalarType>  SparseGaussianLogLikelihoodType;
    typedef typename SparseGaussianLogLikelihoodType::Pointer       SparseGaussianLogLikelihoodTypePointer;


    typedef gpr::GaussianKernel<double, gpr::KernelParameterType::Exponential>             GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

    SparseGaussianProcessInference(KernelTypePointer kernel, TScalarType jitter, TScalarType noise, TScalarType stepwidth, unsigned iterations, unsigned maxnuminducingsamples) :
        m_Jitter(jitter), m_Noise(noise), m_StepWidth(stepwidth), m_NumberOfIterations(iterations), m_MaxNumberOfInducingSamples(maxnuminducingsamples){
        sgp = SparseGaussianProcessTypePointer(new SparseGaussianProcessType(kernel));
        sgp->SetSigma(m_Noise);
    }

    ~SparseGaussianProcessInference(){}

    void AddSamples(const VectorListType& x_samples, const VectorListType& y_samples){
        if(x_samples.size() != y_samples.size()) throw std::string("SparseGaussianProcessInference::AddSamples: number of x and y samples must be equal.");

        for(unsigned i=0; i<x_samples.size(); i++){
            sgp->AddSample(x_samples[i], y_samples[i]);
            m_SampleVectors.push_back(x_samples[i]);
            m_LabelVectors.push_back(y_samples[i]);
        }
    }


    void Optimize(){
        SparseGaussianLogLikelihoodTypePointer sgl(new SparseGaussianLogLikelihoodType());
        sgp->ClearInducingSamples();

        //sgl->DebugOn();

        // build randomized index vector
        std::vector<unsigned> indices;
        for(unsigned i=0; i<sgp->GetNumberOfSamples(); i++){
            indices.push_back(i);
        }
        std::random_shuffle ( indices.begin(), indices.end() );

        ParameterVectorType param_strings = sgp->GetKernel()->GetParameters();
        std::vector<double> parameters;
        for(ParameterType s : param_strings){
            double p;
            std::stringstream ss;
            ss << s;
            ss >> p;
            parameters.push_back(p);
        }

        // successively add inducing sample
        //double current_iterations = -5;
        std::cout << "Add inducing points: " << std::flush;
        for(unsigned i=0; i<std::min(sgp->GetNumberOfSamples(), m_MaxNumberOfInducingSamples); i++){
            std::cout << indices[i] << ", " << std::endl;
            sgp->AddInducingSample(m_SampleVectors[indices[i]], m_LabelVectors[indices[i]]);

            if(!(i%5==0)) continue;
        //}
        //std::cout << std::endl;


            std::cout << "number of samples: " << sgp->GetNumberOfInducingSamples() <<  std::endl;

            // optimize parameters
            //std::cout << "do " << std::exp(-current_iterations) << " iterations " << std::endl;
            for(unsigned iter=0; iter<m_NumberOfIterations; iter++){
                try{
                    GaussianKernelTypePointer gk(new GaussianKernelType(parameters[0], parameters[1]));
                    sgp->SetKernel(gk);

                    std::pair<VectorType, VectorType> value_derivative = sgl->GetValueAndParameterDerivatives(sgp);

                    VectorType lh = value_derivative.first;
                    std::cout << "Likelihood " << lh << ",\t sigma/scale " << std::exp(parameters[0]) << "/" << std::exp(parameters[1]) << std::flush;


                    VectorType likelihood_gradient = value_derivative.second;
                    std::cout << ", Gradients: " << likelihood_gradient.adjoint() << std::flush;

                    if(std::isnan(lh.sum())){
                        throw std::string(" nan in likelihood");
                    }

                    //(likelihood_update.adjoint()*likelihood_update).inverse() * likelihood_update.dot(lh)




                    //std::cout << ", step: " << std::endl << (likelihood_gradient * likelihood_gradient.adjoint()) << std::endl;// * likelihood_gradient * lh << std::endl;


                    //VectorType parameter_update = (pinv<MatrixType>(likelihood_gradient * likelihood_gradient.adjoint()) * likelihood_gradient);
                    VectorType parameter_update = (pinv<MatrixType>(likelihood_gradient * likelihood_gradient.adjoint()) * likelihood_gradient);
                    std::cout << ", inv(J'J)J': " << parameter_update.adjoint() << std::flush;

                    std::cout << ", update: " << std::flush;
                    for(unsigned p=0; p<parameters.size(); p++){
                        TScalarType u;
                        if(parameter_update[p]==0){
                        //std::cout << "log gradient... " << std::endl;
                            if(likelihood_gradient[p]>=0){
                                u = m_StepWidth*m_StepWidth*m_StepWidth*std::log(1+likelihood_gradient[p]);
                            }
                            else{
                                u = -m_StepWidth*m_StepWidth*m_StepWidth*std::log(1+std::fabs(likelihood_gradient[p]));
                            }
                            parameters[p] += u;
                            std::cout << u << " " << std::flush;
                        }
                        else{
                            //std::cout << "gauss newton... " << std::endl;
                            u = parameter_update[p]*lh[0];
                            if(u>0){
                                 u = m_StepWidth*std::log(1+u);
                            }
                            else{
                                u = -m_StepWidth*std::log(1+std::fabs(u));
                            }
                            parameters[p] -= u;
                            std::cout << u << " " << std::flush;
                        }
                    }
                    std::cout << std::flush;


                    std::cout << ", new parameters: " << std::flush;
                    for(unsigned p=0; p<parameters.size(); p++){
                        std::cout << std::exp(parameters[p]) << ", " << std::flush;
                    }
                    std::cout << std::endl;
                }
                catch(std::string& s){
                    std::cout << "[failed] " << s << std::endl;
                    return;
                }
            }
            //current_iterations += 0.1;
        } // add inducing samples
    }

private:
    TScalarType m_Jitter;
    TScalarType m_Noise;
    TScalarType m_StepWidth;
    unsigned m_NumberOfIterations;
    unsigned m_MaxNumberOfInducingSamples;

    SparseGaussianProcessTypePointer sgp;

    VectorListType m_SampleVectors;  // Dimensionality: TInputDimension
    VectorListType m_LabelVectors;   // Dimensionality: TOutputDimension

    SparseGaussianProcessInference(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

// setup sparse Gaussian process
typedef gpr::SparseGaussianProcess<double> SparseGaussianProcessType;
typedef gpr::GaussianProcess<double> GaussianProcessType;
typedef SparseGaussianProcessType::VectorType VectorType;
typedef SparseGaussianProcessType::MatrixType MatrixType;
typedef SparseGaussianProcessType::DiagMatrixType DiagMatrixType;
typedef SparseGaussianProcessType::VectorListType VectorListType;

typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

void Test1(){
    //Eigen::setNbThreads(1);

    std::cout.precision(8);

    unsigned n = 10;

    double noise = 0.02;
    double jitter = 0.011;
    double step = 1e-1;
    //double step = 1e-10;
    unsigned iterations = 500;
    unsigned num_inducing_points = 20;

    // setup kernel and gaussian processes
//    double sigma = 0.23;
//    double scale = 10;
    double sigma = 1;
    double scale = 1;


    typedef SparseGaussianProcessInference<double>                  SparseGaussianProcessInferenceType;
    typedef typename SparseGaussianProcessInferenceType::Pointer    SparseGaussianProcessInferenceTypePointer;

    GaussianKernelTypePointer gk(new GaussianKernelType(sigma, scale));
    SparseGaussianProcessInferenceTypePointer sgpi(new SparseGaussianProcessInferenceType(gk, jitter, noise, step, iterations, num_inducing_points));


    // setup dense gaussian process
    GaussianProcessType::Pointer gp(new GaussianProcessType(gk));
    //gp->DebugOn();
    gp->SetSigma(noise);



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

    // fill up samples
    VectorListType x_samples;
    VectorListType y_samples;
    for(unsigned i=0; i<n; i++){
        x_samples.push_back(VectorType::Constant(1,Xn[i]));
        y_samples.push_back(VectorType::Constant(1,Yn[i]));
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,Yn[i]));
    }
    sgpi->AddSamples(x_samples, y_samples);


    // Sparse maximum likelihood
    //std::cout << "Sparse GP" << std::endl;
    //sgpi->Optimize();

    //return;

    // Dense maximum likelihood
    //std::cout << "Dense GP" << std::endl;
    {
        typedef gpr::GaussianLogLikelihood<double> GaussianLogLikelihoodType;
        typedef std::shared_ptr<GaussianLogLikelihoodType> GaussianLogLikelihoodTypePointer;
        GaussianLogLikelihoodTypePointer gl(new GaussianLogLikelihoodType());
        //std::cout << "Dense likelihood optimization with n=" << gp->GetNumberOfSamples() << " samples" << std::endl;

        double step = 1e-1;
        std::vector<double> parameters;
        parameters.push_back(sigma);
        parameters.push_back(scale);

        for(unsigned i=0; i<iterations; i++){
            // analytical
            try{
                typedef gpr::GaussianKernel<double, gpr::KernelParameterType::Exponential>             GaussianKernelType;
                typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

                GaussianKernelTypePointer gk(new GaussianKernelType(parameters[0], parameters[1]));
                gp->SetKernel(gk);
                //gp->DebugOn();

                std::pair<VectorType, VectorType> value_derivative = gl->GetValueAndParameterDerivatives(gp);
                VectorType likelihood_gradient = value_derivative.second;
                VectorType lh = value_derivative.first;

                std::cout << "Likelihood " << value_derivative.first << ", sigma/scale " << std::exp(parameters[0]) << "/" << std::exp(parameters[1]) << std::flush;

                std::cout << ", Gradients: " << likelihood_gradient.adjoint() << std::flush;

                VectorType parameter_update = (pinv<MatrixType>(likelihood_gradient * likelihood_gradient.adjoint()) * likelihood_gradient);

                std::cout << ", inf(J'J)J': " << parameter_update.adjoint() << std::flush;

//                sigma += lambda * likelihood_gradient[0];
//                scale += lambda * likelihood_gradient[1];





                std::cout << ", update: " << std::flush;
                for(unsigned p=0; p<parameters.size(); p++){
                    double u;
                    if(parameter_update[p]==0){
                    //std::cout << "log gradient... " << std::endl;
                        if(likelihood_gradient[p]>=0){
                            u = step*step*step*std::log(1+likelihood_gradient[p]);
                        }
                        else{
                            u = -step*step*step*std::log(1+std::fabs(likelihood_gradient[p]));
                        }
                        parameters[p] += u;
                        std::cout << u << " " << std::flush;
                    }
                    else{
                        //std::cout << "gauss newton... " << std::endl;
                        u = parameter_update[p]*lh[0];
                        if(u>0){
                             u = step*std::log(1+u);
                        }
                        else{
                            u = -step*std::log(1+std::fabs(u));
                        }
                        parameters[p] -= u;
                        std::cout << u << " " << std::flush;
                    }
                }
                std::cout << std::flush;

                std::cout << ", new parameters: " << std::flush;
                for(unsigned p=0; p<parameters.size(); p++){
                    std::cout << std::exp(parameters[p]) << ", " << std::flush;
                }
                std::cout << std::endl;

            }
            catch(std::string& s){
                std::cout << "[failed] " << s << std::endl;
                return;
            }
        }

        //std::cout << "Likelihood after optimization: " <<  (*gl)(gp) << std::endl;
        //std::cout << "New sigma/scale: " << sigma << "/" << scale << std::endl;
    }



    //sgl->DebugOn();


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


// test the efficient inversion resp. determinant
void Test2(){
    std::cout << "Test 2.1 efficient inversion: ... " << std::flush;

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
        std::cout << "[failed], with an error of " << inv_error << std::endl;
    }

    std::cout << "Test 2.2 efficient determinant: ... " << std::flush;
    //---------------------------------------------------------------------------
    // determinant test
    double determinant = sgl->EfficientDeterminant(I_sigma, K_inv, K, Knm);
    double det_err = std::fabs(determinant-(MatrixType(I_sigma)+Knm*K_inv*Knm.adjoint()).determinant());

    if(det_err < 1e-10){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }

}

void Test3(double jitter){
    std::cout << "Test 3.1 core matrix test (jitter=" << jitter << "): ... " << std::flush;

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

void Test4(){
    std::cout << "Test 4: sparse likelihood gradient test... " << std::flush;

    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.2;
    double jitter = 0.01;

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);


    double h = 0.001;
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

            //std::cout <<  "central difference (sigma): " << d << ", but is " << D[0] <<   std::endl;
            if(std::fabs(D[0] - d) > 1) passed = false;
        }
        {
            // central differences scale
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma, scale+h/2)));
            double plus = (*sgl)(sgp)[0];
            sgp->SetKernel(GaussianKernelTypePointer(new GaussianKernelType(sigma, scale-h/2)));
            double minus = (*sgl)(sgp)[0];

            double d = (plus - minus)/h;

            //std::cout <<  "central difference (scale): " << d << ", but is " << D[1] <<   std::endl;
            if(std::fabs(D[1] - d) > 0.1) passed = false;
        }
    }

    if(passed){
        std::cout << "[passed]" << std::endl;
    }
    else{
        std::cout << "[failed]" << std::endl;
    }

    return;
}

int main (int argc, char *argv[]){
    //std::cout << "Sparse Gaussian Process test: " << std::endl;
    try{
        Test1();
        //Test2();
        //Test3(0); // jitter 0
        //Test3(0.001); // jitter 0.001
        //Test4();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
