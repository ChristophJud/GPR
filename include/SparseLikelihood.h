
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

#include <string>
#include <cmath>
#include <climits>
#include <memory>
#include <chrono>

#include "GaussianProcess.h"
#include "Likelihood.h"
#include "SparseGaussianProcess.h"

#include <Eigen/Dense>

namespace gpr{
template <class TScalarType>
class SparseLikelihood : public Likelihood<TScalarType>{
public:
    typedef SparseLikelihood Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef SparseGaussianProcess<TScalarType> SparseGaussianProcessType;
    typedef typename SparseGaussianProcessType::Pointer SparseGaussianProcessTypePointer;
    typedef typename SparseGaussianProcessType::VectorType VectorType;
    typedef typename SparseGaussianProcessType::MatrixType MatrixType;
    typedef typename SparseGaussianProcessType::DiagMatrixType DiagMatrixType;

    typedef GaussianProcess<TScalarType> GaussianProcessType;
    typedef typename GaussianProcess<TScalarType>::Pointer GaussianProcessTypePointer;

    virtual inline VectorType operator()(const SparseGaussianProcessTypePointer gp) const{
        throw std::string("Likelihood: operator() is not implemented.");
    }

    virtual std::string ToString() const = 0;

    virtual void DebugOn(){
        debug = true;
    }

    SparseLikelihood(): debug(false) { }
    virtual ~SparseLikelihood() {}

    // this method is public for the us in testing
    virtual void GetCoreMatrices(const SparseGaussianProcessTypePointer gp,
                                  MatrixType &K,
                                  MatrixType &K_inv,
                                  MatrixType &Kmn,
                                  DiagMatrixType &I_sigma) const{
        gp->ComputeCoreMatrices(K, K_inv, Kmn, I_sigma);
    }

    // this method is public for the us in testing
    virtual void GetCoreMatrix(const SparseGaussianProcessTypePointer gp, MatrixType& C, MatrixType& K_inv, MatrixType &Kmn) const{
        gp->ComputeCoreMatrix(C, K_inv, Kmn); // computes core
    }

    // this method is public for the us in testing
    virtual TScalarType GetKernelMatrixTrace(const SparseGaussianProcessTypePointer gp) const{
        return gp->ComputeKernelMatrixTrace();
    }

    // this method is public for the us in testing
    virtual VectorType GetDerivativeKernelMatrixTrace(const SparseGaussianProcessTypePointer gp) const{
        return gp->ComputeDerivativeKernelMatrixTrace();
    }

protected:
    // methods using friendship to gaussian process class
    virtual void GetLabelMatrix(const SparseGaussianProcessTypePointer gp, MatrixType& Y) const{
        gp->ComputeDenseLabelMatrix(Y); // get dense labels
    }

    virtual void GetDerivativeKernelMatrix(const SparseGaussianProcessTypePointer gp, MatrixType& D) const{
        gp->ComputeDerivativeKernelMatrix(D); // computes derivative of kernel
    }

    virtual void GetDerivativeKernelVectorMatrix(const SparseGaussianProcessTypePointer gp, MatrixType& D) const{
            gp->ComputeDerivativeKernelVectorMatrix(D);
    }

    virtual MatrixType GetInverseMatrix(const SparseGaussianProcessTypePointer gp, MatrixType &K) const{
        return gp->InvertKernelMatrix(K, gp->GetInversionMethod());
    }

    bool debug;
private:
      SparseLikelihood (const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented
};


template <class TScalarType>
class SparseGaussianLogLikelihood : public SparseLikelihood<TScalarType>{
public:

    typedef SparseLikelihood<TScalarType> Superclass;
    typedef SparseGaussianLogLikelihood Self;
    typedef std::shared_ptr<Self> Pointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::MatrixType MatrixType;
    typedef typename Superclass::DiagMatrixType DiagMatrixType;
    typedef typename Superclass::SparseGaussianProcessTypePointer SparseGaussianProcessTypePointer;
    typedef typename Superclass::GaussianProcessTypePointer GaussianProcessTypePointer;

    // efficient matrix inversion of the form
    //  (A + XBX') = inv(A) - inv(A)X inv(inv(B)+X'inv(A)X) X' inv(A)
    // (here A is a diagonal)
    virtual inline void EfficientInversion(const SparseGaussianProcessTypePointer gp, MatrixType& D, const DiagMatrixType& A, const MatrixType& B, const MatrixType& B_inv, const MatrixType &X) const{
        DiagMatrixType A_inv = (1.0/A.diagonal().array()).matrix().asDiagonal();
        MatrixType C = (B_inv + X.adjoint() * A_inv * X);
        C = this->GetInverseMatrix(gp, C);

        D = MatrixType(A_inv) - A_inv * X * C * X.adjoint() * A_inv;
    }

    // efficient determinant of the form
    // | A + XBX'| = |B|*|A|*|inv(B) + X' inv(A) X|
    virtual inline TScalarType EfficientDeterminant(const DiagMatrixType& A, const MatrixType& B, const MatrixType& B_inv, const MatrixType &X) const{
        DiagMatrixType A_inv = (1.0/A.diagonal().array()).matrix().asDiagonal();

        TScalarType det_B = B.determinant();
        if(std::isinf(det_B)){
            det_B = std::numeric_limits<TScalarType>::max();
        }
        return det_B * A.diagonal().array().prod() * (B_inv + X.adjoint() * A_inv * X).determinant();
    }

    virtual inline VectorType operator()(const SparseGaussianProcessTypePointer gp) const{

        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(gp, K, K_inv, Knm, I_sigma);

        MatrixType Y;
        this->GetLabelMatrix(gp, Y);

        // data fit term
        MatrixType C_inv;
        EfficientInversion(gp, C_inv, I_sigma, K_inv, K, Knm);
        VectorType df = -0.5 * (Y.adjoint() * C_inv * Y);


        // complexity penalty (parameter regularizer)
        TScalarType determinant = this->EfficientDeterminant(I_sigma, K_inv, K, Knm);
        if(determinant < -std::numeric_limits<double>::epsilon()){
            std::stringstream ss;
            ss << "SparseGaussianLogLikelihood: determinant of K is smaller than zero: " << determinant;
            throw ss.str();
        }
        TScalarType cp;

        if(determinant <= 0){
            cp = -0.5 * std::log(std::numeric_limits<double>::min());
        }
        else{
            cp = -0.5 * std::log(determinant);
        }


        // inducing samples regularizer
        MatrixType C;
        this->GetCoreMatrix(gp, C, K_inv, Knm);

        TScalarType Knn_trace = this->GetKernelMatrixTrace(gp);

        TScalarType sr = -0.5/gp->GetSigmaSquared() * (Knn_trace - C.trace());


        // constant term
        TScalarType ct = -gp->GetNumberOfSamples()/2.0 * std::log(2*M_PI);

        if(this->debug){
            std::cout << "Knn trace: " << Knn_trace << std::endl;
            std::cout << "C trace: " << C.trace() << std::endl;
            std::cout << "Data fit: " << df << std::endl;
            std::cout << "Complexity: " << cp << std::endl;
            std::cout << "Constant: " << ct << std::endl;
            std::cout << "Sample regularization: " << sr << std::endl;
        }

        return df.array() + (cp + ct + sr);
    }

    virtual inline VectorType GetParameterDerivatives(const SparseGaussianProcessTypePointer gp) const{

        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(gp, K, K_inv, Knm, I_sigma);

        MatrixType Y;
        this->GetLabelMatrix(gp, Y);

        VectorType Knn_d_trace = this->GetDerivativeKernelMatrixTrace(gp);


        //----------------------------------------------------
        // data fit term
        MatrixType C_inv;
        EfficientInversion(gp, C_inv, I_sigma, K_inv, K, Knm); // inv(I_sigma + Knm inv(Kmm) Kmn)

        // compute: -0.5 Y' inv(C) grad(C) inv(C) Y
        MatrixType Kmm_d;
        this->GetDerivativeKernelMatrix(gp, Kmm_d);

        MatrixType Knm_d;
        this->GetDerivativeKernelVectorMatrix(gp, Knm_d);

        unsigned num_params = gp->GetKernel()->GetNumberOfParameters();
        unsigned n = gp->GetNumberOfSamples();
        unsigned m = gp->GetNumberOfInducingSamples();

        MatrixType A; // grad(C)
        A.resize(num_params*n, n);
        for(unsigned p=0; p<num_params; p++){
            A.block(p*n, 0, n, n) = Knm_d.block(p*n, 0, n, m) * K_inv * Knm.adjoint() -
                                    Knm * K_inv * Kmm_d.block(p*m, 0, m, m) * K_inv * Knm.adjoint() +
                                    Knm * K_inv * Knm_d.block(p*n, 0, n, m).adjoint();
        }

        VectorType dt =  VectorType::Zero(num_params);
        for(unsigned p=0; p<num_params; p++){
            VectorType dTheta_i = 0.5*Y.adjoint()*C_inv*A.block(p*n, 0, n, n)*C_inv*Y;
            if(dTheta_i.rows() != 1) throw std::string("SparseLikelihood::GetParameterDerivatives: dimension missmatch in calculating derivative of data fit term");
            dt[p] = dTheta_i[0];
        }

        //----------------------------------------------------
        // complexity term
        VectorType ct =  VectorType::Zero(num_params);
        for(unsigned p=0; p<num_params; p++){
            TScalarType dTheta_i = -0.5 * (C_inv * A.block(p*n, 0, n, n)).trace();
            ct[p] = dTheta_i;
        }

        //----------------------------------------------------
        // sample regularization term
        VectorType sr =  VectorType::Zero(num_params);
        for(unsigned p=0; p<num_params; p++){
            TScalarType dTheta_i = -0.5/gp->GetSigmaSquared() * (Knn_d_trace[p] - A.block(p*n, 0, n, n).trace());
            sr[p] = dTheta_i;
        }

        return dt + ct + sr;
    }

    SparseGaussianLogLikelihood() : Superclass(){  }
    virtual ~SparseGaussianLogLikelihood() {}

    virtual std::string ToString() const{ return "SparseGaussianLogLikelihood"; }

private:
    SparseGaussianLogLikelihood(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
}
