
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
    typedef typename Superclass::SparseGaussianProcessType SparseGaussianProcessType;
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

    virtual inline VectorType operator()(const GaussianProcessTypePointer gp) const{
        SparseGaussianProcessTypePointer sgp = CastToSparseGaussianProcess(gp);
        if(sgp->GetNumberOfInducingSamples() == 0) throw std::string("SparseLikelihood::GetValueAndParameterDerivative: there are no inducing samples specified");

        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(sgp, K, K_inv, Knm, I_sigma);

        MatrixType Y;
        this->GetLabelMatrix(sgp, Y);

        // data fit term
        MatrixType C_inv;
        EfficientInversion(sgp, C_inv, I_sigma, K_inv, K, Knm);
        VectorType df = -0.5 * (Y.adjoint() * C_inv * Y);


        // complexity penalty (parameter regularizer)
        TScalarType determinant = this->EfficientDeterminant(I_sigma, K_inv, K, Knm);
        TScalarType cp;

        if(determinant <= std::numeric_limits<double>::min()){
            cp = -0.5 * std::log(std::numeric_limits<double>::min());
        }
        else if(determinant > std::numeric_limits<double>::max()){
            cp = -0.5 * std::log(std::numeric_limits<double>::max());
        }
        else{
            cp = -0.5 * std::log(determinant);
        }


        // inducing samples regularizer
        MatrixType C;
        this->GetCoreMatrix(sgp, C, K_inv, Knm);

        TScalarType Knn_trace = this->GetKernelMatrixTrace(sgp);

//        TScalarType sr = -0.5/gp->GetSigmaSquared() * (Knn_trace - C.trace());


        // constant term
        TScalarType ct = -(sgp->GetNumberOfSamples()/2.0) * std::log(2*M_PI);

        if(this->debug){
            std::cout << "Knn trace: " << Knn_trace << std::endl;
            std::cout << "C trace: " << C.trace() << std::endl;
            std::cout << "Data fit: " << df << std::endl;
            std::cout << "Complexity: " << cp << std::endl;
            std::cout << "Constant: " << ct << std::endl;
//            std::cout << "Sample regularization: " << sr << std::endl;
        }

//        return df.array() + (cp + ct + sr);

        VectorType values = df.array() + (cp + ct);
        if(std::isnan(values.sum())){
            throw std::string("SparseLikelihood::GetValueAndParameterDerivative: likelihood value is not a number.");
        }

        return values;
    }

    virtual inline VectorType GetParameterDerivatives(const GaussianProcessTypePointer gp) const{
        SparseGaussianProcessTypePointer sgp = CastToSparseGaussianProcess(gp);
        if(sgp->GetNumberOfInducingSamples() == 0) throw std::string("SparseLikelihood::GetValueAndParameterDerivative: there are no inducing samples specified");

        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(sgp, K, K_inv, Knm, I_sigma);

        MatrixType Y;
        this->GetLabelMatrix(sgp, Y);

//        VectorType Knn_d_trace = this->GetDerivativeKernelMatrixTrace(gp);


        //----------------------------------------------------
        // data fit term
        MatrixType C_inv;
        EfficientInversion(sgp, C_inv, I_sigma, K_inv, K, Knm); // inv(I_sigma + Knm inv(Kmm) Kmn)

        // compute: -0.5 Y' inv(C) grad(C) inv(C) Y
        MatrixType Kmm_d;
        this->GetDerivativeKernelMatrix(sgp, Kmm_d);

        MatrixType Knm_d;
        this->GetDerivativeKernelVectorMatrix(sgp, Knm_d);

        unsigned num_params = sgp->GetKernel()->GetNumberOfParameters();
        unsigned n = sgp->GetNumberOfSamples();
        unsigned m = sgp->GetNumberOfInducingSamples();

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

//        //----------------------------------------------------
//        // sample regularization term
//        VectorType sr =  VectorType::Zero(num_params);
//        for(unsigned p=0; p<num_params; p++){
//            TScalarType dTheta_i = -0.5/gp->GetSigmaSquared() * (Knn_d_trace[p] - A.block(p*n, 0, n, n).trace());
//            sr[p] = dTheta_i;
//        }
//        return dt + ct + sr;
        return dt + ct;
    }

    virtual inline std::pair<VectorType, VectorType> GetValueAndParameterDerivatives(const GaussianProcessTypePointer gp) const{
        SparseGaussianProcessTypePointer sgp = CastToSparseGaussianProcess(gp);
        if(sgp->GetNumberOfInducingSamples() == 0) throw std::string("SparseLikelihood::GetValueAndParameterDerivative: there are no inducing samples specified");

        //------------------------------------
        // get all the important matrices
        MatrixType K;
        MatrixType K_inv;
        MatrixType Knm;
        DiagMatrixType I_sigma;

        this->GetCoreMatrices(sgp, K, K_inv, Knm, I_sigma);


//        std::cout << "Kernel matrix: " << std::endl;
//        std::cout << K << std::endl;
//        std::cout << "Kernel matrix inverted: " << std::endl;
//        std::cout << K_inv << std::endl;

        MatrixType Y;
        this->GetLabelMatrix(sgp, Y);

//        VectorType Knn_d_trace = this->GetDerivativeKernelMatrixTrace(gp);

        MatrixType C_inv;
        EfficientInversion(sgp, C_inv, I_sigma, K_inv, K, Knm); // inv(I_sigma + Knm inv(Kmm) Kmn)

        //----------------------------------------------------
        // data fit term
        VectorType df_value = -0.5 * (Y.adjoint() * C_inv * Y);

        // data fit gradient
        // compute: -0.5 Y' inv(C) grad(C) inv(C) Y
        MatrixType Kmm_d;
        this->GetDerivativeKernelMatrix(sgp, Kmm_d);

        MatrixType Knm_d;
        this->GetDerivativeKernelVectorMatrix(sgp, Knm_d);

        unsigned num_params = sgp->GetKernel()->GetNumberOfParameters();
        unsigned n = sgp->GetNumberOfSamples();
        unsigned m = sgp->GetNumberOfInducingSamples();

        MatrixType A; // grad(C)
        A.resize(num_params*n, n);
        for(unsigned p=0; p<num_params; p++){
            A.block(p*n, 0, n, n) = Knm_d.block(p*n, 0, n, m) * K_inv * Knm.adjoint() -
                                    Knm * K_inv * Kmm_d.block(p*m, 0, m, m) * K_inv * Knm.adjoint() +
                                    Knm * K_inv * Knm_d.block(p*n, 0, n, m).adjoint();
        }

        VectorType df_grad =  VectorType::Zero(num_params);
        for(unsigned p=0; p<num_params; p++){
            VectorType dTheta_i = 0.5*Y.adjoint()*C_inv*A.block(p*n, 0, n, n)*C_inv*Y;
            if(dTheta_i.rows() != 1) throw std::string("SparseLikelihood::GetParameterDerivatives: dimension missmatch in calculating derivative of data fit term");
            df_grad[p] = dTheta_i[0];
        }


        //----------------------------------------------------
        // complexity penalty (parameter regularizer)
        TScalarType determinant = this->EfficientDeterminant(I_sigma, K_inv, K, Knm);
        TScalarType cp_value;

        if(determinant <= std::numeric_limits<double>::min() || std::isnan(determinant)){
            cp_value = -0.5 * std::log(std::numeric_limits<double>::min());
        }
        else if(determinant > std::numeric_limits<double>::max()){
            cp_value = -0.5 * std::log(std::numeric_limits<double>::max());
        }
        else{
            cp_value = -0.5 * std::log(determinant);
        }

        // complexity penalty derivative
        VectorType cp_grad =  VectorType::Zero(num_params);
        for(unsigned p=0; p<num_params; p++){
            TScalarType dTheta_i = -0.5 * (C_inv * A.block(p*n, 0, n, n)).trace();
            cp_grad[p] = dTheta_i;
        }

        //----------------------------------------------------
        // inducing sample regularization term
        MatrixType C;
        this->GetCoreMatrix(sgp, C, K_inv, Knm);

//        TScalarType Knn_trace = this->GetKernelMatrixTrace(gp);

//        TScalarType sr_value = -0.5/gp->GetSigmaSquared() * (Knn_trace - C.trace());
        //std::cout << "Knn_trace: " << Knn_trace << ", C.trace " << C.trace() << std::endl;

//        // inducing sample reg term derivative
//        VectorType sr_grad =  VectorType::Zero(num_params);
//        for(unsigned p=0; p<num_params; p++){
//            TScalarType dTheta_i = -0.5/gp->GetSigmaSquared() * (Knn_d_trace[p] - A.block(p*n, 0, n, n).trace());
//            sr_grad[p] = dTheta_i;
//        }

        //----------------------------------------------------
        // constant term
        TScalarType ct_value = -(sgp->GetNumberOfSamples()/2.0) * std::log(2*M_PI);


        // full value
//        VectorType values = df_value.array() + (cp_value + ct_value + sr_value);
        VectorType values = df_value.array() + (cp_value + ct_value);

//        if(std::isinf(values[0]) || std::isnan(values.sum())){
//            std::cout << "df: " << df_value << ", cp: " << cp_value << ", ct: " << ct_value << ", det " << determinant << std::endl;
//        }

        if(std::isnan(values.sum())){
            throw std::string("SparseLikelihood::GetValueAndParameterDerivative: likelihood value is not a number.");
        }

        // full gradient
//        VectorType derivatives = df_grad + cp_grad + sr_grad;
        VectorType derivatives = df_grad + cp_grad;

        //std::cout << "df: " << df_grad.adjoint() << ", cp: " << cp_grad.adjoint() << ", " << sr_grad.adjoint() << std::endl;

        return std::make_pair(values, derivatives);
    }

    SparseGaussianLogLikelihood() : Superclass(){  }
    virtual ~SparseGaussianLogLikelihood() {}

    virtual std::string ToString() const{ return "SparseGaussianLogLikelihood"; }

private:
    SparseGaussianProcessTypePointer CastToSparseGaussianProcess(const GaussianProcessTypePointer gp) const{
        SparseGaussianProcessTypePointer sgp(std::dynamic_pointer_cast<SparseGaussianProcessType>(gp));
        if(sgp.get()==NULL) throw std::string("SparseGaussianLogLikelihood: cannot cast to SparseGaussianProcess");
        return sgp;
    }

    SparseGaussianLogLikelihood(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
}
