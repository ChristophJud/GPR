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

#include <cmath>
#include <string>
#include <complex>
#include <random>
#include <ctime>

namespace gpr {

// Superclass for density functions
// all densities have to be evaluatable
template<class TScalarType>
class Density{
public:

    virtual TScalarType operator()(TScalarType x) = 0;
    Density(){}
    virtual ~Density(){}

    virtual TScalarType mean() = 0;
    virtual TScalarType variance() = 0;

    static std::default_random_engine g;
};
template<class TScalarType>
std::default_random_engine Density<TScalarType>::g(static_cast<unsigned int>(std::time(0)));


/*
 * Gaussian Density: to evaluate the distribution at a point x
 *
 * p(x|mu,sigma) =
 *
 */
template<class TScalarType>
class GaussianDensity : public Density<TScalarType>{
public:
    GaussianDensity(TScalarType mu, TScalarType sigma) : mu(mu), sigma(sigma) {
        if(sigma<=0) throw std::string("GaussianDensity: the Gaussian density is only defined for sigma>0");
        normal_dist = std::normal_distribution<TScalarType>(mu, sigma);
    }
    ~GaussianDensity(){}
    TScalarType operator()(TScalarType x){
        return 1/(sigma*std::sqrt(2*M_PI)) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
    }

    TScalarType operator()(){
        return normal_dist(Density<TScalarType>::g);
    }

    TScalarType GetDerivative(TScalarType x){
        return -(x-mu) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) / (std::sqrt(2)*std::sqrt(M_PI)*sigma*sigma*sigma);
    }

    TScalarType mean(){
        return mu;
    }

    TScalarType variance(){
        return sigma;
    }

private:
    TScalarType mu;
    TScalarType sigma;
    std::normal_distribution<TScalarType> normal_dist;
};


/*
 * Inverse Gaussian Density: to evaluate the distribution at a point x
 *
 * p(x|lambda,mu) = (lambda/(2pi x^3)^(0.5) exp(-lambda(x-mu)^2/(2mu^2 x))
 *
 */
template<class TScalarType>
class InverseGaussianDensity : public Density<TScalarType>{
public:
    InverseGaussianDensity(TScalarType lambda, TScalarType mu) : lambda(lambda), mu(mu) {
        if(lambda<=0 || mu<=0) throw std::string("InverseGaussianDensity: the inverse Gaussian density is only defined for lambda>0 and mu>0");
        normal_dist = std::normal_distribution<TScalarType>(0, 1);
        uniform_dist = std::uniform_real_distribution<TScalarType>(0, 1);
    }
    ~InverseGaussianDensity(){}
    TScalarType operator()(TScalarType x){
        if(x<=0) throw std::string("InverseGaussianDensity: domain error. The inverse Gaussian density is not defined for x<=0.");

        return std::sqrt(lambda/(2*M_PI*x*x*x)) * std::exp(-lambda*(x-mu)*(x-mu)/(2*mu*mu*x));
    }

    TScalarType operator()(){
        double v = normal_dist(Density<TScalarType>::g);
        double y = v*v;
        double x = mu+(mu*mu*y)/(2*lambda) - mu/(2*lambda) * std::sqrt(4*mu*lambda*y + (mu*mu)*(y*y));
        double z = uniform_dist(Density<TScalarType>::g);
        if(z <= mu/(mu+x)){
            return x;
        }
        else{
            return mu*mu/x;
        }
    }

    TScalarType mean(){
        return mu;
    }

    TScalarType variance(){
        return mu*mu*mu/lambda;
    }

private:
    TScalarType lambda;
    TScalarType mu;
    std::normal_distribution<TScalarType> normal_dist;
    std::uniform_real_distribution<TScalarType> uniform_dist;
};


/*
 * Gamma Density: to evaluate the distribution at a point x
 *
 * p(x|alpha,beta) = 1/(beta^alpha * gamma(alpha)) * x^{alpha-1} * exp(-x/beta)
 *
 */
template<class TScalarType>
class GammaDensity : public Density<TScalarType>{
public:
    GammaDensity(TScalarType alpha, TScalarType beta) : alpha(alpha), beta(beta) {
        if(alpha<=0 || beta<=0) throw std::string("GammaDensity: the Gamma density is only defined for alpha>0 and beta>0");
        factor = std::pow(beta,alpha) * std::tgamma(alpha);
        distribution = std::gamma_distribution<TScalarType>(alpha, beta);
    }
    ~GammaDensity(){}
    TScalarType operator()(TScalarType x){
        if(x==0 && alpha < 1) throw std::string("GammaDensity: domain error. For alpha < 1 and x==0 the Gamma density is not defined.");
        if(x<0) throw std::string("GammaDensity: domain error. The Gamma density is not defined for x<0.");

        // compute log power
        std::complex<TScalarType> cx(x);
        std::complex<TScalarType> alphamin1(alpha-1);

        return std::abs(std::pow(cx,alphamin1)) / factor * std::exp(-x/beta);
    }

    TScalarType operator()(){
        return distribution(Density<TScalarType>::g);
    }

    TScalarType mean(){
        return alpha/beta;
    }

    TScalarType variance(){
        return alpha/(beta*beta);
    }

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
    std::gamma_distribution<TScalarType> distribution;
};

}
