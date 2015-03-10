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

namespace gpr {

// Superclass for density functions
// all densities have to be evaluatable
template<class TScalarType>
class Density{
public:

    virtual TScalarType operator()(TScalarType x) = 0;
    Density(){}
    virtual ~Density(){}


    static std::default_random_engine g;
};
template<class TScalarType>
std::default_random_engine Density<TScalarType>::g;


/*
 * Gaussian Density: to evaluate the distribution at a point x
 *
 * p(x|lambda,mu) = (lambda/(2pi x^3)^(0.5) exp(-lambda(x-mu)^2/(2mu^2 x))
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

private:
    TScalarType mu;
    TScalarType sigma;
    std::normal_distribution<TScalarType> normal_dist;
};

/*
 * Log Gaussian Density: to evaluate the distribution at a point x
 *
 * p(x|lambda,mu) = (lambda/(2pi x^3)^(0.5) exp(-lambda(x-mu)^2/(2mu^2 x))
 *
 */
template<class TScalarType>
class LogGaussianDensity : public Density<TScalarType>{
public:
    LogGaussianDensity(TScalarType mu, TScalarType sigma) : mu(mu), sigma(sigma) {
        if(sigma<=0) throw std::string("LogGaussianDensity: the log Gaussian density is only defined for sigma>0");
        dist = std::lognormal_distribution<TScalarType>(mu, sigma);
    }
    ~LogGaussianDensity(){}
    TScalarType operator()(TScalarType x){
        if(x<=0) throw std::string("LogGaussianDensity: the log Gaussian density is only defined for x > 0");
        return 1/(x*sigma*std::sqrt(2*M_PI)) * std::exp(-(std::log(x)-mu)*(std::log(x)-mu)/(2*sigma*sigma));
    }

    TScalarType operator()(){
        return dist(Density<TScalarType>::g);
    }

    TScalarType GetDerivative(TScalarType x){
        if(x<=0) throw std::string("LogGaussianDensity: the derivative of the log Gaussian density is only defined for x > 0");
        double f1 = std::exp(-(std::log(x)-mu)*(std::log(x)-mu)/(2*sigma*sigma));
        return -f1 * (std::log(x)-mu) / (std::sqrt(2)*std::sqrt(M_PI)*sigma*sigma*sigma*x*x) - f1 / (std::sqrt(2)*std::sqrt(M_PI)*sigma*x*x);
    }

private:
    TScalarType mu;
    TScalarType sigma;
    std::lognormal_distribution<TScalarType> dist;
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

private:
    TScalarType lambda;
    TScalarType mu;
    std::normal_distribution<TScalarType> normal_dist;
    std::uniform_real_distribution<TScalarType> uniform_dist;
};

/*
 * Log Inverse Gaussian Density: needed for maximum likelihood
 *
 * p(x|lambda,mu) = (lambda/(2pi x^3)^(0.5) exp(-lambda(x-mu)^2/(2mu^2 x))
 *
 */
template<class TScalarType>
class LogInverseGaussianDensity : public Density<TScalarType>{
public:
    LogInverseGaussianDensity(TScalarType lambda, TScalarType mu) : lambda(lambda), mu(mu) {
        if(lambda<=0 || mu<=0) throw std::string("InverseGaussianDensity: the inverse Gaussian density is only defined for lambda>0 and mu>0");
    }
    ~LogInverseGaussianDensity(){}
    TScalarType operator()(TScalarType x){
        if(x<=0) throw std::string("LogInverseGaussianDensity: domain error. The inverse Gaussian density is not defined for x<=0.");
        if(x==1) throw std::string("LogInverseGaussianDensity: domain error. The log inverse Gaussian density is not defined for x==1.");
        double logx = std::log(x);
        return std::abs(1/x) * std::sqrt(1/(2*M_PI*logx*logx*logx)) * std::exp(-lambda*(logx-mu)*(logx-mu)/(2*mu*mu*logx));
    }

private:
    TScalarType lambda;
    TScalarType mu;
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

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
    std::gamma_distribution<TScalarType> distribution;
};

/*
 * Log Gamma Density: to evaluate the distribution at a point x
 *
 * p(x|alpha,beta) = 1/(beta^alpha * gamma(alpha)) * log(x)^{alpha-1} * exp(-log(x)/beta)
 *
 */
template<class TScalarType>
class LogGammaDensity : public Density<TScalarType>{
public:
    LogGammaDensity(TScalarType alpha, TScalarType beta) : alpha(alpha), beta(beta) {
        if(alpha<=0 || beta<=0) throw std::string("LogGammaDensity: the log Gamma density is only defined for alpha>0 and beta>0");
        factor = std::pow(beta,alpha) * std::tgamma(alpha);
    }
    ~LogGammaDensity(){}
    TScalarType operator()(TScalarType x){
        if(x==1 && alpha<1) throw std::string("LogGammaDensity: domain error. For alpha<1 and x==0 the log Gamma density is not defined.");
        if(x<=0) throw std::string("LogGammaDensity: domain error. The log Gamma density is not defined for x<=0.");

        // compute log power
        std::complex<TScalarType> logx(std::log(x));
        std::complex<TScalarType> alphamin1(alpha-1);

        return std::abs(std::pow(logx,alphamin1)) / (x*factor) * std::exp(-std::log(x)/beta);
    }
    TScalarType GetDerivative(TScalarType x){
        if(x==1 && alpha<2) throw std::string("LogGammaDensity: domain error. For alpha<2 and x==0 the derivative of log Gamma density is not defined.");
        if(x<=0) throw std::string("LogGammaDensity: domain error. The log Gamma density is not defined for x<=0.");

        // compute log power
        std::complex<TScalarType> logx(std::log(x));
        std::complex<TScalarType> alphamin1(alpha-1);
        std::complex<TScalarType> alphamin2(alpha-2);

        TScalarType f1 = std::abs(std::pow(logx,alphamin1));
        TScalarType f2 = std::abs(std::pow(logx,alphamin2));

        TScalarType summand1 = (alpha-1)*std::exp(-std::log(x)/beta)*f2 / (factor*x*x);
        TScalarType summand2 = -std::pow(beta,-alpha-1)*std::exp(-std::log(x)/beta)*f1 / (std::tgamma(alpha)*x*x);
        TScalarType summand3 = -std::exp(-std::log(x)/beta)*f1/(factor*x*x);

        return summand1 + summand2 + summand3;
    }

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
};

}
