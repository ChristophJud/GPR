/*
 * Copyright 2015 Christoph Jud (christoph.jud@unibas.ch)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#pragma once

#include <limits>

#include "Kernel.h"
#include "GaussianProcess.h"


typedef gpr::GaussianKernel<double>             GaussianKernelType;
typedef std::shared_ptr<GaussianKernelType>     GaussianKernelTypePointer;

namespace gpr{
template <class TScalarType> class SparseLikelihood;

template< class TScalarType >
class SparseGaussianProcess : public GaussianProcess<TScalarType>{
public:
    typedef SparseGaussianProcess Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef GaussianProcess<TScalarType>  Superclass;
    typedef typename Superclass::VectorType         VectorType;
    typedef typename Superclass::MatrixType         MatrixType;
    typedef typename Superclass::DiagMatrixType     DiagMatrixType;
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

    // this method is public for using it in testing
    virtual void ComputeDenseKernelMatrix(MatrixType &M) const{
        if(this->debug){
            std::cout << "SparseGaussianProcess::ComputeDenseKernelMatrix: building kernel matrix... ";
            std::cout.flush();
        }

        Superclass::ComputeKernelMatrixInternal(M, this->m_SampleVectors);

        if(this->debug) std::cout << "[done]" << std::endl;
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

    virtual void ComputeKernelMatrixWithJitter(MatrixType &M) const{
        ComputeKernelMatrix(M);
        // add jitter to diagonal
        for(unsigned i=0; i<M.rows(); i++){
            M(i,i) += m_Jitter;
        }
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
     * Bring the dense label vectors in a matrix form Y
     *
     * (calls the superclass method)
     */
    virtual void ComputeDenseLabelMatrix(MatrixType &Y) const{
        Superclass::ComputeLabelMatrix(Y);
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
     * Computation of the cross-covariance matrix Kmn = k(x_i, y_j)
     * where x is in the inducing samples and y in the dense samples
     *
     *  - Kmn = [Kx1 Kx2 ... Kxm] in R^nxm
     *
     * (calls ComputeKernelVectorInternal)
     */
    virtual void ComputeKernelVectorMatrix(MatrixType &Knm) const{

        unsigned n = this->m_SampleVectors.size();
        unsigned m = m_InducingSampleVectors.size();

        if(!(m<=n)){
            throw std::string("SparseGaussianProcess::ComputeKernelVectorMatrix: number of dense samples must be higher than the number of sparse samples");
        }

        Knm.resize(n, m);

#pragma omp parallel for
        for(unsigned i=0; i<n; i++){
            for(unsigned j=0; j<m; j++){
                Knm(i, j) = (*this->m_Kernel)(m_InducingSampleVectors[j], this->m_SampleVectors[i]);
            }
        }
    }

    virtual void ComputeDerivativeKernelVectorMatrix(MatrixType &M)const{
        unsigned num_params = this->m_Kernel->GetNumberOfParameters();

        unsigned n = this->m_SampleVectors.size();
        unsigned m = m_InducingSampleVectors.size();

        if(!(m<=n)){
            throw std::string("SparseGaussianProcess::ComputeDerivativeKernelVectorMatrix: number of dense samples must be higher than the number of sparse samples");
        }
        M.resize(n*num_params,m);

    #pragma omp parallel for
        for(unsigned i=0; i<n; i++){
            for(unsigned j=0; j<m; j++){
                typename GaussianProcess<TScalarType>::VectorType v;
                v = this->m_Kernel->GetDerivative(m_InducingSampleVectors[j], this->m_SampleVectors[i]);

                if(v.rows() != num_params) throw std::string("GaussianProcess::ComputeDerivativeKernelMatrixInternal: dimension missmatch in derivative.");
                for(unsigned p=0; p<num_params; p++){

                    //if(i+p*n >= M.rows() || j+p*n >= M.rows())  throw std::string("GaussianProcess::ComputeDerivativeKernelMatrix: dimension missmatch in derivative.");

                    M(i + p*n, j) = v[p];
                    //M(j + p*n, i) = v[p];

                    //if(i + p*n == j) M(i + p*n, j) += m_Sigma; // TODO: not sure if this is needed
                }
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
        bool stable = (m_Jitter<std::numeric_limits<TScalarType>::min())? true : false;

        MatrixType K;
        ComputeKernelMatrixWithJitter(K);

        // inverting inducing kernel matrix
        m_IndusingInvertedKernelMatrix = this->InvertKernelMatrix(K, this->m_InvMethod, stable);

        // computing kernel vector matrix between inducing points and dense points
        MatrixType Kmn;
        ComputeKernelVectorMatrix(Kmn);

        // Computing label matrix
        // calculate label matrix
        // TODO: if a mean support is implemented, the mean has to be subtracted from the labels!
        MatrixType Y;
        ComputeDenseLabelMatrix(Y);

        // computation of Sigma matrix
        TScalarType inverse_sigma2 = 1.0/(this->m_Sigma*this->m_Sigma);
        MatrixType S = K + inverse_sigma2*Kmn.adjoint()*Kmn;
        m_SigmaMatrix = this->InvertKernelMatrix(S, this->m_InvMethod, stable);

        // regression vectors for computing mean
        m_RegressionVectors = m_IndusingInvertedKernelMatrix * (inverse_sigma2*K*m_SigmaMatrix*Kmn.adjoint()*Y);

        // regression matrix for computing variance
        m_RegressionMatrix = m_IndusingInvertedKernelMatrix * (K*m_SigmaMatrix*K) * m_IndusingInvertedKernelMatrix;

        // core matrix, used for likelihoods
        MatrixType C;
        ComputeCoreMatrix(C, m_IndusingInvertedKernelMatrix, Kmn);
        m_CoreMatrix = C;

    }


    /*
     * Computation of the following:
     *  - Inducing kernel matrix K
     *  - Inducing inverted kernel matrix inv(K)
     *  - Cross kernel matrix Kmn
     *  - Identity (noise) matrix I_sigma
     */
    virtual void ComputeCoreMatrices(MatrixType &K, MatrixType &K_inv, MatrixType &Kmn, DiagMatrixType &I_sigma){
        bool stable = (m_Jitter<std::numeric_limits<TScalarType>::min())? true : false;

        if(this->debug) std::cout << "SparseGaussianProcess::ComputeCoreMatrices: compute kernel matrix..." << std::flush;
        ComputeKernelMatrixWithJitter(K);
        if(this->debug) std::cout << " [done]" << std::endl;

        if(this->debug) std::cout << "SparseGaussianProcess::ComputeCoreMatrices: invert kernel matrix..." << std::flush;
        K_inv = this->InvertKernelMatrix(K, this->m_InvMethod, stable);
        if(this->debug) std::cout << " [done]" << std::endl;

        if(this->debug) std::cout << "SparseGaussianProcess::ComputeCoreMatrices: compute link kernel matrix..." << std::flush;
        ComputeKernelVectorMatrix(Kmn);
        if(this->debug) std::cout << " [done]" << std::endl;

        if(this->GetSigma()<=0){
            throw std::string("SparseGaussianProcess::ComputeCoreMatrices: sigma must be positive.");
        }
        if(Kmn.rows() == 0){
            throw std::string("SparseGaussianProcess::ComputeCoreMatrices: empty sample set.");
        }

        if(this->debug) std::cout << "SparseGaussianProcess::ComputeCoreMatrices: compute additional noise..." << std::flush;
        I_sigma.resize(Kmn.rows());
        I_sigma.setIdentity();
        I_sigma = (I_sigma.diagonal().array() * this->GetSigma()).matrix().asDiagonal();
        if(this->debug) std::cout << " [done]" << std::endl;
    }

    /*
     * Computation of core matrix: Kmn * inv(Kmm) * Knm
     */
    virtual void ComputeCoreMatrix(MatrixType &C, MatrixType &K_inv) const{
        bool stable = (m_Jitter<std::numeric_limits<TScalarType>::min())? true : false;

        MatrixType K;
        ComputeKernelMatrixWithJitter(K);

        MatrixType Kmn;
        ComputeKernelVectorMatrix(Kmn);

        std::cout << "C " << Kmn.rows() << " x " << Kmn.cols() << std::endl;

        K_inv = this->InvertKernelMatrix(K, this->m_InvMethod, stable);

        ComputeCoreMatrix(C, K_inv, Kmn);
    }

    // additional interfaces for ComputeCoreMatrix
    virtual void ComputeCoreMatrix(MatrixType &C) const{
        MatrixType K_inv;
        ComputeCoreMatrix(C, K_inv);
    }
    virtual void ComputeCoreMatrix(MatrixType &C, const MatrixType& K_inv, const MatrixType& Kmn) const{
        C = Kmn * K_inv * Kmn.adjoint();
    }

    /*
     * Computation of the derivative inducing kernel matrix D_i = delta Kmm / delta params_i
     * 	- returns a matrix: [D_0
     *                        .
     *                       D_i
     *                        .
     *                       D_l-1]
     *    for l = number of params and D_i in mxm, m = number of inducing samples
     */
    virtual void ComputeDerivativeKernelMatrix(MatrixType &M) const{
        if(this->debug){
            std::cout << "SparseGaussianProcess::ComputeDerivativeKernelMatrix: building kernel matrix... ";
            std::cout.flush();
        }
        Superclass::ComputeDerivativeKernelMatrixInternal(M, m_InducingSampleVectors);

        if(this->debug) std::cout << "[done]" << std::endl;
    }


    TScalarType m_Jitter; // noise on inducing kernel matrix
    bool m_Initialized;

    VectorListType m_InducingSampleVectors;  // Dimensionality: TInputDimension
    VectorListType m_InducingLabelVectors;  // Dimensionality: TOutputDimension
    MatrixType m_RegressionVectors;         // mu of m(x)
    MatrixType m_SigmaMatrix;
    MatrixType m_IndusingInvertedKernelMatrix;
    MatrixType m_RegressionMatrix;
    MatrixType m_CoreMatrix;    // Knm * inv(Kmm) * Kmn

private:
    SparseGaussianProcess(const Self &); //purposely not implemented
    void operator=(const Self &); //purposely not implemented

    friend class SparseLikelihood<TScalarType>;
};

} // namespace gpr

#include "SparseLikelihood.h"
