
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
        MatrixType D;
        EfficientInversion(gp, D, I_sigma, K_inv, K, Knm);
        VectorType df = -0.5 * (Y.adjoint() * D * Y);


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

        TScalarType sr = -0.5/gp->GetSigma() * (Knn_trace - C.trace());


        // constant term
        TScalarType ct = -C.rows()/2.0 * std::log(2*M_PI);

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
        std::cout << "computing parameter derivatives ... " << std::endl;

        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(gp, K, K_inv, Knm, I_sigma);

        std::cout << "inv(Kmm)" << std::endl;
        std::cout << K_inv << std::endl;

        std::cout << "Kmn" << std::endl;
        std::cout << Knm << std::endl;


        MatrixType Y;
        this->GetLabelMatrix(gp, Y);

        // data fit term
        std::cout << "data fit" << std::endl;
        MatrixType C_inv;
        EfficientInversion(gp, C_inv, I_sigma, K_inv, K, Knm); // inv(I_sigma + Knm inv(Kmm) Kmn)

        // compute: -0.5 Y' inv(C) grad(C) inv(C) Y

        std::cout << "inducing kernel derivative matrix" << std::endl;
        // D has the dimensions of num_params*C.rows x C.cols
        MatrixType Kmm_d;
        this->GetDerivativeKernelMatrix(gp, Kmm_d);

        std::cout << "derivative kmm" << std::endl;
        std::cout << Kmm_d << std::endl;

        MatrixType Knm_d;
        this->GetDerivativeKernelVectorMatrix(gp, Knm_d);

        std::cout << "derivative kmn" << std::endl;
        std::cout << Knm_d << std::endl;

        MatrixType A; // derivative of I_sigma + Knm inv(Kmm) Kmn
        A = Knm_d*K_inv*Knm.adjoint() + Knm*K_inv*Kmm_d*K_inv*Knm.adjoint() + Knm * K_inv * Knm_d.adjoint();

        std::cout << "first term" << std::endl;
        std::cout << Knm_d*K_inv*Knm.adjoint() << std::endl;

        std::cout << "second term" << std::endl;
        std::cout << Knm*K_inv*Kmm_d*K_inv*Knm.adjoint() << std::endl;

        std::cout << "third term" << std::endl;
        std::cout << Knm * K_inv * Knm_d.adjoint() << std::endl;


        std::cout << "derivative A" << std::endl;
        std::cout << A << std::endl;


//        std::cout << "stuff" << std::endl;
//        unsigned num_params = static_cast<unsigned>(D.rows()/D.cols());
//        if(static_cast<double>(D.rows())/static_cast<double>(D.cols()) - num_params != 0){
//            throw std::string("SparseGaussianLogLikelihood: wrong dimension of derivative kernel matrix.");
//        }
        VectorType delta = VectorType::Zero(1);


//        for(unsigned p=0; p<num_params; p++){
//            delta[p] = 0.5 * ((alpha*alpha.adjoint() - C) * D.block(p*D.cols(),0,D.cols(),D.cols())).trace();
//        }

        return delta;
    }

    SparseGaussianLogLikelihood() : Superclass(){  }
    virtual ~SparseGaussianLogLikelihood() {}

    virtual std::string ToString() const{ return "SparseGaussianLogLikelihood"; }

private:
    SparseGaussianLogLikelihood(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
}
