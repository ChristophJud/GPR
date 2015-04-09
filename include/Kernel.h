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
#include <vector>
#include <map>
#include <cmath>
#include <memory>
#include <iomanip>

#include <Eigen/Dense>

#include "MatrixIO.h"

namespace gpr{

template <class TScalarType> class KernelFactory;

/*
 * Kernel interface. Operator () has to be implemented in subclass
 */
template <class TScalarType>
class Kernel{
public:

	typedef Kernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
	typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;
    typedef std::string ParameterType;
    typedef std::vector<ParameterType> ParameterVectorType;


    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        throw std::string("Kernel: operator() is not implemented.");
    }

    // can be calculated with http://www.derivative-calculator.net/
    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        throw std::string("Kernel: GetDerivative() is not implemented.");
    }

    virtual std::string ToString() const = 0;
    virtual unsigned GetNumberOfParameters() const = 0;

    // returns the parameter vector
    virtual inline const ParameterVectorType GetParameters(){
        return m_parameters;
	}

    std::string ParametersToString(const ParameterVectorType& params) const{
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<TScalarType>::digits10 +1);
        for(unsigned i=0; i<params.size(); i++){
            ss << params[i] << ",";
        }
        return ss.str();
    }

    // the constructor only registers the kernels
    // at the KernelFactory
    Kernel() {
        KernelFactory<TScalarType>::RegisterKernels();
    }
    virtual ~Kernel() {}


    // Comparison operators (mainly used by the tests)
    virtual bool operator ==(const Kernel<TScalarType> &b) const{
        if(this->ToString() != b.ToString()){
            return false;
        }
        if(this->m_parameters.size() != b.m_parameters.size()){
            return false;
        }
        for(unsigned i=0; i<this->m_parameters.size(); i++){
            if(this->m_parameters[i].compare(b.m_parameters[i])){
                return false;
            }
        }
        return true;
    }
    virtual bool operator !=(const Kernel<TScalarType> &b) const{
        return ! operator ==(b);
    }

protected:
    // in m_parameters, all parameters which should be saved/loaded have to be pushed
    // they are stored as strings such that strings and scalars can be parameters.
    ParameterVectorType m_parameters;

    // scalars to string conversion with maximal precision
    static std::string P2S(TScalarType p){
        std::stringstream ss;
        ss << std::setprecision(std::numeric_limits<TScalarType>::digits10 +1);
        ss << p;
        return ss.str();
    }

    // String to scalar conversion
    static TScalarType S2P(const ParameterType& s){
        TScalarType p;
        std::stringstream ss;
        ss << s;
        ss >> p;
        return p;
    }

private:
	  Kernel(const Self&); //purposely not implemented
	  void operator=(const Self&); //purposely not implemented
};


/*
 * Sum Kernel: k(x,y) = k1(x,y) + k2(x,y)
 */
template <class TScalarType>
class SumKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef typename Superclass::SelfPointer SuperclassPointer;
    typedef SumKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterType ParameterType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        return (*m_Kernel1)(x, y) + (*m_Kernel2)(x, y);
    }

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D1 = m_Kernel1->GetDerivative(x,y);
        VectorType D2 = m_Kernel2->GetDerivative(x,y);
        VectorType D = VectorType::Zero(D1.rows() + D2.rows());

        D.topRows(D1.rows()) = D1;
        D.bottomRows(D2.rows()) = D2;

        return D;
    }

    // Constructor takes two kernel pointers
    // If the parameters are available, it is better to Load
    // the SumKernel from the KernelFactory
    SumKernel(SuperclassPointer const k1, SuperclassPointer const k2) : Superclass(),
        k_string1(k1->ToString()),
        k_string2(k2->ToString()),
        num_params1(k1->GetParameters().size()),
        num_params2(k2->GetParameters().size()),
        m_Kernel1(k1),
        m_Kernel2(k2){

        // store kernel strings
        this->m_parameters.push_back(k_string1);
        this->m_parameters.push_back(k_string2);

        // store number of parameters per kernel
        // needed to load the full kernel string
        this->m_parameters.push_back(Self::P2S(num_params1)); // as strings
        this->m_parameters.push_back(Self::P2S(num_params2));

        // store both parameter vectors of kernel 1 and kernel 2
        for(ParameterType p : k1->GetParameters()){
            this->m_parameters.push_back(p);
        }
        for(ParameterType p : k2->GetParameters()){
            this->m_parameters.push_back(p);
        }
    }
    virtual ~SumKernel() {}

    virtual std::string ToString() const{
        return "SumKernel("+m_Kernel1->ToString()+","+m_Kernel2->ToString()+")";
    }

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() < 4){
            throw std::string("SumKernel::Load: wrong number of kernel parameters.");
        }

        // read kernel strings
        std::string ks1 = parameters[0];
        std::string ks2 = parameters[1];

        // read number of parameters for k1 and k2
        unsigned np1 = Self::S2P(parameters[2]);
        unsigned np2 = Self::S2P(parameters[3]);

        // read the parameter vectors into
        // two separate parameter vectors
        ParameterVectorType params1;
        ParameterVectorType params2;

        for(unsigned i=4; i<4+np1; i++){
            params1.push_back(parameters[i]); // fill up parameters for first kernel
        }

        for(unsigned i=4+np1; i<4+np1+np2; i++){
            params2.push_back(parameters[i]); // fill up parameters for second kernel
        }

        // return a sum kernel where the summands are loaded from the kernel factory
        return SelfPointer(new Self(KernelFactory<TScalarType>::Load(ks1, params1),
                        KernelFactory<TScalarType>::Load(ks2, params2)));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_Kernel1->GetNumberOfParameters() + m_Kernel2->GetNumberOfParameters();
    }

    const SuperclassPointer GetKernel1() { return m_Kernel1; }
    const SuperclassPointer GetKernel2() { return m_Kernel2; }

private:
    const std::string k_string1;
    const std::string k_string2;
    const unsigned num_params1;
    const unsigned num_params2;
    const SuperclassPointer m_Kernel1;
    const SuperclassPointer m_Kernel2;

    SumKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

/*
 * Product Kernel: k(x,y) = k1(x,y) * k2(x,y)
 */
template <class TScalarType>
class ProductKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef typename Superclass::SelfPointer SuperclassPointer;
    typedef ProductKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterType ParameterType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        return (*m_Kernel1)(x, y) * (*m_Kernel2)(x, y);
    }

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D1 = m_Kernel1->GetDerivative(x,y);
        VectorType D2 = m_Kernel2->GetDerivative(x,y);
        VectorType D = VectorType::Zero(D1.rows() + D2.rows());

        D.topRows(D1.rows()) = D1 * (*m_Kernel2)(x,y);
        D.bottomRows(D2.rows()) = D2 * (*m_Kernel1)(x,y);

        return D;
    }

    // Constructor takes two kernel pointers
    // If the parameters are available, it is better to Load
    // the ProductKernel from the KernelFactory
    ProductKernel(SuperclassPointer const k1, SuperclassPointer const k2) : Superclass(),
        k_string1(k1->ToString()),
        k_string2(k2->ToString()),
        num_params1(k1->GetParameters().size()),
        num_params2(k2->GetParameters().size()),
        m_Kernel1(k1),
        m_Kernel2(k2){

        // store kernel strings
        this->m_parameters.push_back(k_string1);
        this->m_parameters.push_back(k_string2);

        // store number of parameters per kernel
        // needed to load the full kernel string
        this->m_parameters.push_back(Self::P2S(num_params1)); // as strings
        this->m_parameters.push_back(Self::P2S(num_params2));

        // store both parameter vectors of kernel 1 and kernel 2
        for(ParameterType p : k1->GetParameters()){
            this->m_parameters.push_back(p);
        }
        for(ParameterType p : k2->GetParameters()){
            this->m_parameters.push_back(p);
        }
    }
    virtual ~ProductKernel() {}

    virtual std::string ToString() const{
        return "ProductKernel("+m_Kernel1->ToString()+","+m_Kernel2->ToString()+")";
    }

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() < 4){
            throw std::string("ProductKernel::Load: wrong number of kernel parameters.");
        }

        // read kernel strings
        std::string ks1 = parameters[0];
        std::string ks2 = parameters[1];

        // read number of parameters for k1 and k2
        unsigned np1 = Self::S2P(parameters[2]);
        unsigned np2 = Self::S2P(parameters[3]);

        // read the parameter vectors into
        // two separate parameter vectors
        ParameterVectorType params1;
        ParameterVectorType params2;

        for(unsigned i=4; i<4+np1; i++){
            params1.push_back(parameters[i]); // fill up parameters for first kernel
        }

        for(unsigned i=4+np1; i<4+np1+np2; i++){
            params2.push_back(parameters[i]); // fill up parameters for second kernel
        }

        // return a sum kernel where the summands are loaded from the kernel factory
        return SelfPointer(new Self(KernelFactory<TScalarType>::Load(ks1, params1),
                        KernelFactory<TScalarType>::Load(ks2, params2)));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_Kernel1->GetNumberOfParameters() + m_Kernel2->GetNumberOfParameters();
    }

    const SuperclassPointer GetKernel1() { return m_Kernel1; }
    const SuperclassPointer GetKernel2() { return m_Kernel2; }

private:
    const std::string k_string1;
    const std::string k_string2;
    const unsigned num_params1;
    const unsigned num_params2;
    const SuperclassPointer m_Kernel1;
    const SuperclassPointer m_Kernel2;

    ProductKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};


/*
 * Gaussian Kernel: k(x,y) = scale^2 * exp( -0.5||x-y||^2 / sigma^2 )
 *
 * - sigma is a smoothness parameter
 * - scale is the expected amplitude
 */
template <class TScalarType>
class GaussianKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef GaussianKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        TScalarType r = (x-y).norm();
        return m_Scale2 * std::exp(-0.5 * (r*r) / (m_Sigma2));
	}

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D = VectorType::Zero(2);

        TScalarType r = (x-y).norm();
        TScalarType f = std::exp(-0.5 * (r*r) / (m_Sigma2));
        D[0] = m_Scale2 * (r*r) / (m_Sigma3) * f; // to sigma
        D[1] = 2*m_Scale * f; // to scale
        return D;
    }

    // for convenience the constructor can be called
    // with scalars or with strings (ParameterType)
    GaussianKernel(TScalarType sigma, TScalarType scale=1) : Superclass(),
            m_Sigma(sigma),
            m_Scale(scale),
            m_Sigma2(sigma*sigma),
            m_Sigma3(sigma*sigma*sigma),
            m_Scale2(scale*scale){

        this->m_parameters.push_back(Self::P2S(m_Sigma));
        this->m_parameters.push_back(Self::P2S(m_Scale));
	}
    GaussianKernel(const typename Superclass::ParameterType& p1,
                   const typename Superclass::ParameterType& p2) :
        GaussianKernel(Self::S2P(p1), Self::S2P(p2)){
    }

	virtual ~GaussianKernel() {}

    virtual std::string ToString() const{
        return "GaussianKernel("+Superclass::ParametersToString(this->m_parameters)+")";
    } // needed for identification

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() != m_NumberOfParameters){
            throw std::string("GaussianKernel::Load: wrong number of kernel parameters.");
        }
        return SelfPointer(new Self(Self::S2P(parameters[0]), Self::S2P(parameters[1])));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_NumberOfParameters;
    }

private:
    TScalarType m_Sigma;
    TScalarType m_Scale;

    TScalarType m_Sigma2;
    TScalarType m_Sigma3;
    TScalarType m_Scale2;

    static unsigned m_NumberOfParameters;
	
	GaussianKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
template <class TScalarType>
unsigned GaussianKernel<TScalarType>::m_NumberOfParameters = 2;


/*
 * White Kernel: k(x,y) = scale^2 * delta
 *
 * - scale is the expected amplitude
 * - only not zero if x==y
 */
template <class TScalarType>
class WhiteKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef WhiteKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        if((x-y).norm() == 0){
            return m_Scale2;
        }
        else{
            return 0;
        }
    }

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D = VectorType::Zero(1);
        if((x-y).norm() == 0){
            D[0] = 2*m_Scale;
        }
        else{
            D[0] = 0;
        }
        return D;
    }

    // for convenience the constructor can be called
    // with scalars or with strings (ParameterType)
    WhiteKernel(TScalarType scale) : Superclass(),
            m_Scale(scale),
            m_Scale2(scale*scale){
        this->m_parameters.push_back(Self::P2S(scale));
    }
    WhiteKernel(const typename Superclass::ParameterType& p1) :
        WhiteKernel(Self::S2P(p1)){
    }

    virtual ~WhiteKernel() {}

    virtual std::string ToString() const{
        return "WhiteKernel("+Superclass::ParametersToString(this->m_parameters)+")";
    } // needed for identification

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() != m_NumberOfParameters){
            throw std::string("GaussianKernel::Load: wrong number of kernel parameters.");
        }
        return SelfPointer(new Self(Self::S2P(parameters[0])));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_NumberOfParameters;
    }

private:
    TScalarType m_Scale;
    TScalarType m_Scale2;

    static unsigned m_NumberOfParameters;

    WhiteKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
template <class TScalarType>
unsigned WhiteKernel<TScalarType>::m_NumberOfParameters = 1;


/*
 * Rational Quadratic Kernel: k(x,y) = scale^2 * pow( 1 + ||x-y||^2 / ( 2 * alpha * sigma^2), -alpha )
 *
 * - sigma is a smoothness parameter
 * - scale is the expected amplitude
 * - alpha is favoring large-scale or small-scale variations (if alpha is inf, it is equal to the gaussian kernel)
 */
template <class TScalarType>
class RationalQuadraticKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef RationalQuadraticKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        TScalarType r = (x-y).norm();
        return m_Scale2 * std::pow(1 + 0.5*(r*r)/(m_Sigma2 * m_Alpha), - m_Alpha);
    }

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D = VectorType::Zero(3);

        TScalarType r = (x-y).norm();
        TScalarType f = 0.5 * r*r / (m_Sigma2*m_Alpha) + 1;
        D[0] = 2*m_Scale * std::pow(f,-m_Alpha);
        D[1] = m_Scale2 * (r*r) * std::pow(f, -m_Alpha - 1) / m_Sigma3;
        D[2] = m_Scale2 *((r*r/(2*m_Sigma2*f*m_Alpha))-std::log(f))*std::pow(f,-m_Alpha);
        return D;
    }

    // for convenience the constructor can be called
    // with scalars or with strings (ParameterType)
    RationalQuadraticKernel(TScalarType scale, TScalarType sigma, TScalarType alpha) : Superclass(),
        m_Scale(scale),
        m_Sigma(sigma),
        m_Alpha(alpha),
        m_Scale2(scale*scale),
        m_Sigma2(sigma*sigma),
        m_Sigma3(sigma*sigma*sigma){

        this->m_parameters.push_back(Self::P2S(m_Scale));
        this->m_parameters.push_back(Self::P2S(m_Sigma));
        this->m_parameters.push_back(Self::P2S(m_Alpha));
    }
    RationalQuadraticKernel(const typename Superclass::ParameterType& p1,
                   const typename Superclass::ParameterType& p2,
                   const typename Superclass::ParameterType& p3) :
        RationalQuadraticKernel(Self::S2P(p1), Self::S2P(p2), Self::S2P(p3)){
    }

    virtual ~RationalQuadraticKernel() {}

    virtual std::string ToString() const{
        return "RationalQuadraticKernel("+Superclass::ParametersToString(this->m_parameters)+")";
    } // needed for identification

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() != m_NumberOfParameters){
            throw std::string("RationalQuadraticKernel::Load: wrong number of kernel parameters.");
        }
        return SelfPointer(new Self(Self::S2P(parameters[0]), Self::S2P(parameters[1]), Self::S2P(parameters[2])));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_NumberOfParameters;
    }

private:
    TScalarType m_Scale;
    TScalarType m_Sigma;
    TScalarType m_Alpha;

    TScalarType m_Scale2;
    TScalarType m_Sigma2;
    TScalarType m_Sigma3;

    static unsigned m_NumberOfParameters;

    RationalQuadraticKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
template <class TScalarType>
unsigned RationalQuadraticKernel<TScalarType>::m_NumberOfParameters = 3;

/*
 * Periodic Kernel: k(x,y) = scale^2 exp( -0.5 sum_d=1^D sin(b(x_d-y_d))/sigma_d)^2 )
 *
 * - D is the number of input dimensions
 * - b is determined as follows: pi/b is the period length
 * - alpha is the expected amplitude
 * - sigma is the smoothnes (more difficult to estimate)
 */
template <class TScalarType>
class PeriodicKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef PeriodicKernel Self;
    typedef std::shared_ptr<Self> SelfPointer;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
        TScalarType sum = 0;
        for(unsigned i=0; i<x.rows(); i++){
            double f = std::sin(m_B*(x[i] - y[i]));
            sum += f*f;
        }

        return m_Scale2 * std::exp(-0.5*sum/m_Sigma2);
    }

    virtual inline VectorType GetDerivative(const VectorType & x, const VectorType & y) const{
        VectorType D = VectorType::Zero(3);

        TScalarType f1 = 0;
        for(unsigned i=0; i<x.rows(); i++){
            double r = std::sin(m_B*(x[i] - y[i]));
            f1 += r*r;
        }

        TScalarType f2 = 0;
        for(unsigned i=0; i<x.rows(); i++){
            double r = (x[i] - y[i]);
            f2 += 2*r * std::cos(m_B*r) * std::sin(m_B*r);
        }

        TScalarType f3 = 0;
        for(unsigned i=0; i<x.rows(); i++){
            double r = (x[i] - y[i]);
            double v = std::sin(m_B*r);
            f3 += (v*v);
        }

        D[0] = 2*m_Scale * std::exp(-0.5*f1/m_Sigma2);
        D[1] = -0.5 * m_Scale2 * std::exp(-0.5*f1/m_Sigma2) * f2 / m_Sigma2;
        D[2] = m_Scale2 * std::exp(-0.5*f1/m_Sigma2) * f3 / m_Sigma3;
        return D;
    }

    PeriodicKernel(TScalarType scale,
                   TScalarType b,
                   TScalarType sigma) : Superclass(),
            m_Scale(scale),
            m_B(b),
            m_Sigma(sigma),
            m_Scale2(scale*scale),
            m_Sigma2(sigma*sigma),
            m_Sigma3(sigma*sigma*sigma)
            {

        this->m_parameters.push_back(Self::P2S(m_Scale));
        this->m_parameters.push_back(Self::P2S(m_B));
        this->m_parameters.push_back(Self::P2S(m_Sigma));
    }
    PeriodicKernel(const typename Superclass::ParameterType& p1,
                   const typename Superclass::ParameterType& p2,
                   const typename Superclass::ParameterType& p3) :
        PeriodicKernel(Self::S2P(p1), Self::S2P(p2), Self::S2P(p3)){

    }
    virtual ~PeriodicKernel() {}

    virtual std::string ToString() const{
        return "PeriodicKernel("+Superclass::ParametersToString(this->m_parameters)+")";
    }

    // Needed from the KernelFactory to instantiate
    // a kernel given a parameter vector;
    static SelfPointer Load(const ParameterVectorType& parameters){
        if(parameters.size() != m_NumberOfParameters){
            throw std::string("PeriodicKernel::Load: wrong number of kernel parameters.");
        }
        return SelfPointer(new Self(Self::S2P(parameters[0]), Self::S2P(parameters[1]), Self::S2P(parameters[2])));
    }

    virtual unsigned GetNumberOfParameters() const{
        return m_NumberOfParameters;
    }

private:
    TScalarType m_Scale;
    TScalarType m_B;
    TScalarType m_Sigma;

    TScalarType m_Scale2;
    TScalarType m_Sigma2;
    TScalarType m_Sigma3;

    static unsigned m_NumberOfParameters;

    PeriodicKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};
template <class TScalarType>
unsigned PeriodicKernel<TScalarType>::m_NumberOfParameters = 3;


}

#include "KernelFactory.h"

