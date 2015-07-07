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

#include "GaussianProcess.h"

#include <Eigen/Dense>

namespace gpr{

/*
 * Kernel interface. Operator () has to be implemented in subclass
 */
template <class TScalarType>
class Likelihood{
public:
    typedef Likelihood Self;
    typedef std::shared_ptr<Self> SelfPointer;

    typedef GaussianProcess<TScalarType> GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorType VectorType;
    typedef typename GaussianProcessType::MatrixType MatrixType;

    virtual inline VectorType operator()(const GaussianProcessTypePointer gp) const{
        throw std::string("Likelihood: operator() is not implemented.");
    }

    virtual std::string ToString() const = 0;

    Likelihood() { }
    virtual ~Likelihood() {}

protected:
    // methods using friendship to gaussian process class
    void GetLabelMatrix(const GaussianProcessTypePointer gp, MatrixType& Y) const{
        gp->ComputeLabelMatrix(Y);
    }

    TScalarType GetCoreMatrix(const GaussianProcessTypePointer gp, MatrixType& C) const{
        return gp->ComputeCoreMatrixWithDeterminant(C); // computes core and returns kernel matrix
    }

    void GetDerivativeKernelMatrix(const GaussianProcessTypePointer gp, MatrixType& D) const{
        return gp->ComputeDerivativeKernelMatrix(D); // computes derivative of kernel
    }

private:
      Likelihood (const Self&); //purposely not implemented
      void operator=(const Self&); //purposely not implemented
};

template <class TScalarType>
class GaussianLikelihood : public Likelihood<TScalarType>{
public:

    typedef Likelihood<TScalarType> Superclass;
    typedef GaussianLikelihood Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::MatrixType MatrixType;
    typedef typename Superclass::GaussianProcessTypePointer GaussianProcessTypePointer;

    virtual inline VectorType operator()(const GaussianProcessTypePointer gp) const{
        MatrixType Y; // label matrix
        this->GetLabelMatrix(gp, Y);

        MatrixType C; // core matrix inv(K + sigmaI)
        TScalarType determinant; // determinant of K + sigma I
        determinant = this->GetCoreMatrix(gp, C);

        // data fit
        VectorType df = -0.5 * (Y.adjoint() * C * Y);
        for(unsigned i=0; i<df.rows(); i++){
            df[i] = std::exp(df[i]);
        }

        // complexity penalty
        if(determinant < -std::numeric_limits<double>::epsilon()){
            std::stringstream ss;
            ss << "GaussianLikelihood: determinant of K is smaller than zero: " << determinant;
            throw ss.str();
        }
        TScalarType cp;

        if(determinant <= 0){
            cp = 1.0/std::sqrt(std::numeric_limits<double>::min());
        }
        else{
            cp = 1.0/std::sqrt(determinant);
        }

        // constant term
        TScalarType ct = 1.0/std::pow(2*M_PI,C.rows()/2.0);

        return df.array() * cp * ct;
    }


    GaussianLikelihood() : Superclass(){  }
    virtual ~GaussianLikelihood() {}

    virtual std::string ToString() const{ return "GaussianLikelihood"; }

private:
    GaussianLikelihood(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

template <class TScalarType>
class GaussianLogLikelihood : public Likelihood<TScalarType>{
public:

    typedef Likelihood<TScalarType> Superclass;
    typedef GaussianLogLikelihood Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::MatrixType MatrixType;
    typedef typename Superclass::GaussianProcessTypePointer GaussianProcessTypePointer;

    virtual inline VectorType operator()(const GaussianProcessTypePointer gp) const{
        MatrixType Y; // label matrix
        this->GetLabelMatrix(gp, Y);

        MatrixType C; // core matrix inv(K + sigmaI)
        TScalarType determinant; // determinant of K + sigma I
        determinant = this->GetCoreMatrix(gp, C);

        // data fit
        VectorType df = -0.5 * (Y.adjoint() * C * Y);

        // complexity penalty
        if(determinant < -std::numeric_limits<double>::epsilon()){
            std::stringstream ss;
            ss << "GaussianLogLikelihood: determinant of K is smaller than zero: " << determinant;
            throw ss.str();
        }
        TScalarType cp;

        if(determinant <= 0){
            cp = -0.5 * std::log(std::numeric_limits<double>::min());
        }
        else{
            cp = -0.5 * std::log(determinant);
        }


        // constant term
        TScalarType ct = -C.rows()/2.0 * std::log(2*M_PI);

        return df.array() + (cp + ct);
    }

    virtual inline VectorType GetParameterDerivatives(const GaussianProcessTypePointer gp) const{
        MatrixType Y; // label matrix
        this->GetLabelMatrix(gp, Y);

        MatrixType C; // core matrix inv(K + sigmaI)
        this->GetCoreMatrix(gp, C);

        MatrixType alpha = C*Y;

        // D has the dimensions of num_params*C.rows x C.cols
        MatrixType D;
        this->GetDerivativeKernelMatrix(gp, D);

        unsigned num_params = static_cast<unsigned>(D.rows()/D.cols());
        if(static_cast<double>(D.rows())/static_cast<double>(D.cols()) - num_params != 0){
            throw std::string("GaussianLogLikelihood: wrong dimension of derivative kernel matrix.");
        }
        VectorType delta = VectorType::Zero(num_params);


        for(unsigned p=0; p<num_params; p++){
            delta[p] = 0.5 * ((alpha*alpha.adjoint() - C) * D.block(p*D.cols(),0,D.cols(),D.cols())).trace();
        }

        return delta;
    }

    GaussianLogLikelihood() : Superclass(){  }
    virtual ~GaussianLogLikelihood() {}

    virtual std::string ToString() const{ return "GaussianLogLikelihood"; }

private:
    GaussianLogLikelihood(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};


}
