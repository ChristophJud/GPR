#ifndef Kernel_h
#define Kernel_h

#include <string>
#include <vector>
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
    typedef std::vector<TScalarType> ParameterVectorType;

	virtual inline TScalarType operator()(const VectorType & x, const VectorType & y){
		throw std::string("Kernel: operator() is not implemented.");
	}

	virtual inline std::string ToString(){
		throw std::string("Kernel: ToString() is not implemented.");
	}

    virtual inline const ParameterVectorType GetParameters(){
        return m_parameters;
	}

	Kernel() {}
	virtual ~Kernel() {}

protected:
    ParameterVectorType m_parameters;
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

	std::string ToString(){ return "GaussianKernel"; }

private:
    TScalarType m_Sigma;
    TScalarType m_Sigma_Squared;
    TScalarType m_Scale;
	
	GaussianKernel(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};


#endif
