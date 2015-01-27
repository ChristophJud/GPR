#ifndef Kernel_h
#define Kernel_h

#include <string>
#include <cmath>
#include <Eigen/Dense>

#include "MatrixIO.h"

/*
 * Kernel interface. Operator () has to be implemented in subclass
 */
template <class TScalarType>
class Kernel{
public:

	typedef Kernel Self;
	
	typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;

	virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
		throw std::string("Kernel: operator() is not implemented.");
	}

	virtual inline std::string ToString(){
		throw std::string("Kernel: ToString() is not implemented.");
	}

	virtual inline TScalarType GetParameter(){
		throw std::string("Kernel: GetParameter() is not implemented.");
	}

	Kernel() {}
	virtual ~Kernel() {}

private:
	  Kernel(const Self&); //purposely not implemented
	  void operator=(const Self&); //purposely not implemented
};

/*
 * Gaussian Kernel: k(x,y) = exp( -||x-y||^2 / sigma^2 )
 */
template <class TScalarType>
class GaussianKernel : public Kernel<TScalarType>{
public:

	typedef GaussianKernel Self;
	
	typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;

    virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
        TScalarType exponent = (x-y).norm()/m_Sigma_Squared;
		return std::exp(-0.5*(exponent*exponent));
	}

    GaussianKernel(TScalarType sigma) : m_Sigma(sigma), m_Sigma_Squared(sigma*sigma) {
		m_Normalization = 1./(sigma*std::sqrt(2*M_PI));
	}
	virtual ~GaussianKernel() {}

	TScalarType GetParameter(){ return m_Sigma; }
	std::string ToString(){ return "GaussianKernel"; }

private:
    TScalarType m_Sigma;
    TScalarType m_Sigma_Squared;
	TScalarType m_Normalization;
	
	GaussianKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};


#endif
