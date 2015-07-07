
#include <string>
#include <iostream>
#include <memory>

#include <boost/random.hpp>

#include <Eigen/Dense>

#include "Kernel.h"
#include "GaussianProcess.h"


typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

template< class TScalarType >
class SparseGaussianProcess : public gpr::GaussianProcess<TScalarType>{
public:
    typedef SparseGaussianProcess Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef typename gpr::GaussianProcess<TScalarType>  Superclass;
    typedef typename Superclass::VectorType         VectorType;
    typedef typename Superclass::MatrixType         MatrixType;
    typedef typename Superclass::VectorListType     VectorListType;
    typedef typename Superclass::MatrixListType     MatrixListType;
    typedef typename Superclass::KernelType         KernelType;
    typedef typename Superclass::KernelTypePointer  KernelTypePointer;


    // Constructors
    SparseGaussianProcess(KernelTypePointer kernel) : Superclass(kernel),
                                                        m_Jitter(0),
                                                        m_Initialized(false){}

    // Destructor
    virtual ~SparseGaussianProcess(){}

    /*
     * Add a new inducing sample lable pair to the sparse Gaussian process
     *  x is the input vector
     *  y the corresponding label vector
     */
    void AddInducingSample(const VectorType& x, const VectorType& y){
        if(m_InducingSampleVectors.size() == 0){ // first call of AddSample defines dimensionality of input space
            this->m_InputDimension = x.size();
        }
        if(m_InducingLabelVectors.size() == 0){ // first call of AddSample defines dimensionality of output space
            this->m_OutputDimension = y.size();
        }

        this->CheckInputDimension(x, "SparseGaussianProcess::AddInducingSample: ");
        this->CheckOutputDimension(y, "SparseGaussianProcess::AddInducingSample: ");

        m_InducingSampleVectors.push_back(x);
        m_InducingLabelVectors.push_back(y);
        m_Initialized = false;
    }

    VectorType Predict(const VectorType &x){
        Initialize();
        this->CheckInputDimension(x, "GaussianProcess::Predict: ");
        VectorType Kx;
        ComputeKernelVector(x, Kx);
        return (Kx.adjoint() * m_RegressionVectors).adjoint();
    }

    TScalarType operator()(const VectorType & x, const VectorType & y){
        Initialize();
        this->CheckInputDimension(x, "SparseGaussianProcess::(): ");
        this->CheckInputDimension(y, "SparseGaussianProcess::(): ");
        VectorType Kx;
        ComputeKernelVector(x, Kx);
        VectorType Ky;
        ComputeKernelVector(y, Ky);

        return (*this->m_Kernel)(x, y) -
                Kx.adjoint() * m_IndusingInvertedKernelMatrix * Ky +
                Kx.adjoint() * m_RegressionMatrix * Ky;
    }

    unsigned GetNumberOfSamples() const{
        return m_InducingSampleVectors.size();
    }

    TScalarType GetJitter() const{
        return m_Jitter;
    }

    void SetJitter(TScalarType jitter){
        m_Jitter = jitter;
        m_Initialized = false;
    }

    virtual void Initialize(){
        if(m_Initialized){
            return;
        }
        if(!(m_InducingSampleVectors.size() > 0)){
            throw std::string("SparseGaussianProcess::Initialize: no inducing samples defined during initialization");
        }
        if(!(m_InducingLabelVectors.size() > 0)){
            throw std::string("SparseGaussianProcess::Initialize: no inducing labels defined during initialization");
        }
        if(!(this->m_SampleVectors.size() > 0)){
            throw std::string("SparseGaussianProcess::Initialize: no dense samples defined during initialization");
        }
        if(!(this->m_LabelVectors.size() > 0)){
            throw std::string("SparseGaussianProcess::Initialize: no dense labels defined during initialization");
        }

        PreComputeRegression();
        m_Initialized = true;

    }

protected:
    /*
     * Computation of inducing kernel matrix K_ij = k(x_i, x_j)
     * 	- it is symmetric therefore only half of the kernel evaluations
     * 	  has to be performed
     *
     * (The actual computation is performed in ComputeKernelMatrixInternal)
     */
    virtual void ComputeKernelMatrix(MatrixType &M) const{
        if(this->debug){
            std::cout << "SparseGaussianProcess::ComputeKernelMatrix: building kernel matrix... ";
            std::cout.flush();
        }

        Superclass::ComputeKernelMatrixInternal(M, m_InducingSampleVectors);

        if(this->debug) std::cout << "[done]" << std::endl;
    }

    /*
     * Bring the label vectors in a matrix form Y,
     * where the rows are the labels.
     *
     * (it is actually performed in ComputeLabelMatrixInternal)
     */
    virtual void ComputeLabelMatrix(MatrixType &Y) const{
        Superclass::ComputeLabelMatrixInternal(Y, m_InducingLabelVectors);
    }

    /*
     * Computation of the kernel vector V_i = k(x, x_i)
     *
     * (calls ComputeKernelVectorInternal)
     */
    virtual void ComputeKernelVector(const VectorType &x, VectorType &Kx) const{
        Superclass::ComputeKernelVectorInternal(x, Kx, m_InducingSampleVectors);
    }

    /*
     * Computation of the kernel vector matrix Kmn = k(x_i, y_j)
     * where x is in the inducing samples and y in the dense samples
     *
     *  - Kmn = [Kx1 Kx2 ... Kxm] in R^nxm
     *
     * (calls ComputeKernelVectorInternal)
     */
    virtual void ComputeKernelVectorMatrix(MatrixType &Kmn) const{

        unsigned n = this->m_SampleVectors.size();
        unsigned m = m_InducingSampleVectors.size();

        if(!(m<=n)){
            throw std::string("SparseGaussianProcess::ComputeKernelVectorMatrix: number of dense samples must be higher than the number of sparse samples");
        }

        Kmn.resize(n, m);

#pragma omp parallel for
        for(unsigned i=0; i<n; i++){
            for(unsigned j=0; j<m; j++){
                Kmn(i, j) = (*this->m_Kernel)(m_InducingSampleVectors[j], this->m_SampleVectors[i]);
            }
        }
    }


    /*
     * Lerning is performed.
     *
     * Mean:
     *  Kxm * inv(Kmm) * mu, mu = sigma^2 Kmm * Sigma * Kmn * Y
     *
     */
    virtual void PreComputeRegression(){
        // Computation of kernel matrix
        if(this->debug){
            std::cout << "SparseGaussianProcess::PreComputeRegression: calculating regression vectors and regression matrix... " << std::endl;
        }

        MatrixType K;
        ComputeKernelMatrix(K);

        // add jitter to diagonal
        for(unsigned i=0; i<K.rows(); i++){
            K(i,i) += m_Jitter;
        }

        // inverting inducing kernel matrix
        m_IndusingInvertedKernelMatrix = this->InvertKernelMatrix(K, this->m_InvMethod);

        // computing kernel vector matrix between inducing points and dense points
        MatrixType Kmn;
        ComputeKernelVectorMatrix(Kmn);

        // Computing label matrix
        // calculate label matrix
        // TODO: if a mean support is implemented, the mean has to be subtracted from the labels!
        MatrixType Y;
        Superclass::ComputeLabelMatrix(Y);

        // computation of Sigma matrix
        TScalarType inverse_sigma2 = 1.0/(this->m_Sigma*this->m_Sigma);
        MatrixType S = K + inverse_sigma2*Kmn.adjoint()*Kmn;
        m_SigmaMatrix = this->InvertKernelMatrix(S, this->m_InvMethod);


        // regression vectors for computing mean
        m_RegressionVectors = m_IndusingInvertedKernelMatrix * (inverse_sigma2*K*m_SigmaMatrix*Kmn.adjoint()*Y);

        // regression matrix for computing variance
        m_RegressionMatrix = m_IndusingInvertedKernelMatrix * (K*m_SigmaMatrix*K) * m_IndusingInvertedKernelMatrix;

    }

    TScalarType m_Jitter; // noise on inducing kernel matrix
    bool m_Initialized;

    VectorListType m_InducingSampleVectors;  // Dimensionality: TInputDimension
    VectorListType m_InducingLabelVectors;  // Dimensionality: TOutputDimension
    MatrixType m_RegressionVectors;         // mu of m(x)
    MatrixType m_SigmaMatrix;
    MatrixType m_IndusingInvertedKernelMatrix;
    MatrixType m_RegressionMatrix;

private:
    SparseGaussianProcess(const Self &); //purposely not implemented
    void operator=(const Self &); //purposely not implemented
};

void Test1(){

    // setup sparse Gaussian process
    typedef SparseGaussianProcess<double> SparseGaussianProcessType;
    typedef gpr::GaussianProcess<double> GaussianProcessType;
    typedef SparseGaussianProcessType::VectorType VectorType;
    typedef SparseGaussianProcessType::MatrixType MatrixType;


    // generate a cool ground truth function
    auto f = [](double x)->double { return (0.5*std::sin(x+10*x) + std::sin(4*x))*x*x; };
    double noise = 0.1;
    double jitter = 0;

    static boost::minstd_rand randgen(static_cast<unsigned>(time(0)));
    static boost::normal_distribution<> dist(0, noise);
    static boost::variate_generator<boost::minstd_rand, boost::normal_distribution<> > r(randgen, dist);


    // setup kernel
    double sigma = 0.1;
    double scale = 1;
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
        gp->AddSample(VectorType::Constant(1,Xn[i]), VectorType::Constant(1,v));
    }

    // small index set
    unsigned m = 13;
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

int main (int argc, char *argv[]){
    //std::cout << "Sparse Gaussian Process test: " << std::endl;
    try{
        Test1();
    }
    catch(std::string& s){
        std::cout << "Error: " << s << std::endl;
    }

    return 0;
}
