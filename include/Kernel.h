#ifndef Kernel_h
#define Kernel_h

#include <string>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include "MatrixIO.h"

namespace gpr{

/*
 * Kernel interface. Operator () has to be implemented in subclass
 */
template <class TScalarType>
class Kernel{
public:

	typedef Kernel Self;
	typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;
    typedef std::vector<TScalarType> ParameterVectorType;

	virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
		throw std::string("Kernel: operator() is not implemented.");
	}

    virtual std::string ToString() const = 0;

    virtual inline const ParameterVectorType GetParameters(){
        return m_parameters;
	}

	Kernel() {}
	virtual ~Kernel() {}

    virtual bool operator !=(const Kernel<TScalarType> &b) const{
        return ! operator ==(b);
    }

    virtual bool operator ==(const Kernel<TScalarType> &b) const{
        if(this->ToString() != b.ToString()){
            return false;
        }
        if(this->m_parameters.size() != b.m_parameters.size()){
            return false;
        }
        for(unsigned i=0; i<this->m_parameters.size(); i++){
            if(this->m_parameters[i] - b.m_parameters[i]>0){
                return false;
            }
        }
        return true;
    }

protected:
    // in m_parameters, all parameters which should be saved/loaded have to be pushed
    ParameterVectorType m_parameters;
private:
	  Kernel(const Self&); //purposely not implemented
	  void operator=(const Self&); //purposely not implemented
};

/*
 * Gaussian Kernel: k(x,y) = exp( -0.5||x-y||^2 / sigma^2 )
 */
template <class TScalarType>
class GaussianKernel : public Kernel<TScalarType>{
public:

    typedef Kernel<TScalarType> Superclass;
    typedef GaussianKernel Self;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
        TScalarType exponent = (x-y).norm()/m_Sigma_Squared;
        return m_Scale * std::exp(-0.5*(exponent*exponent));
	}

    GaussianKernel(TScalarType sigma, TScalarType scale=1) : Superclass(),
            m_Sigma(sigma),
            m_Sigma_Squared(sigma*sigma),
            m_Scale(scale) {

        this->m_parameters.push_back(m_Sigma);
        this->m_parameters.push_back(m_Scale);
	}
	virtual ~GaussianKernel() {}

    virtual std::string ToString() const{ return "GaussianKernel"; }

private:
    TScalarType m_Sigma;
    TScalarType m_Sigma_Squared;
    TScalarType m_Scale;
	
	GaussianKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};


/*
 * Periodic Kernel: k(x,y) = alpha^2 exp( -0.5 sum_d=1^D sin(b(x_d-y_d))/sigma_d)^2 )
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
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::ParameterVectorType ParameterVectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
        TScalarType sum = 0;
        for(unsigned i=0; i<x.rows(); i++){
            double f = std::sin(m_B*(x[i] - y[i]))/m_Sigma;
            sum += f*f;
        }

        return m_Alpha_Squared * std::exp(-0.5*sum);
    }

    PeriodicKernel(TScalarType alpha,
                   TScalarType b,
                   TScalarType sigma) : Superclass(),
            m_Alpha(alpha),
            m_B(b),
            m_Sigma(sigma),
            m_Alpha_Squared(alpha*alpha)
            {

        this->m_parameters.push_back(m_Alpha);
        this->m_parameters.push_back(m_B);
        this->m_parameters.push_back(m_Sigma);
    }
    virtual ~PeriodicKernel() {}

    virtual std::string ToString() const{ return "PeriodicKernel"; }

private:
    TScalarType m_Alpha;
    TScalarType m_B;
    TScalarType m_Sigma;
    TScalarType m_Alpha_Squared;

    PeriodicKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

}

#endif
