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
#include <utility>

#include <boost/math/special_functions/gamma.hpp>

namespace gpr {

// Superclass for density functions
// all densities have to be evaluatable
template<class TScalarType>
class Density{
public:

    typedef std::pair<TScalarType, TScalarType> PairType;

    virtual TScalarType operator()(TScalarType x) const = 0;
    Density(){}
    virtual ~Density(){}

    virtual TScalarType cdf(TScalarType x) const = 0;
    virtual TScalarType mean() const = 0;
    virtual TScalarType variance() const = 0;
    virtual TScalarType mode() const = 0;

    TScalarType icdf(TScalarType y){
        if(y<0 || y>1) throw std::string("Density::icdf: domain error. y is in [0,1]");
        TScalarType s = 0;
        TScalarType step = 1;
        TScalarType old_s = s;

        for(unsigned i=0; i<1000; i++){
            if(this->cdf(s)<y){
                step *= 2;
                s += step;
            }
            else{
                step /= 2;
                s -= step;
            }
            if(std::abs(s-old_s)<1e-6){
                break;
            }
            else{
                old_s = s;
            }
        }

        return reflect(s,this->cdf(s), 1.0, 0.0).second;
    }

protected:
    static std::default_random_engine g;

private:
    // reflection line: y = ax + c
    PairType reflect(TScalarType x, TScalarType y, TScalarType a, TScalarType c){
        TScalarType d = (x + (y-c)*a)/(1+a*a);
        return std::make_pair(2*d-x, 2*d*a -y + 2*c);
    }
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
    TScalarType operator()(TScalarType x) const{
        return 1/(sigma*std::sqrt(2*M_PI)) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
    }

    TScalarType operator()() const{
        return normal_dist(Density<TScalarType>::g);
    }

    TScalarType GetDerivative(TScalarType x) const{
        return -(x-mu) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) / (std::sqrt(2)*std::sqrt(M_PI)*sigma*sigma*sigma);
    }

    TScalarType cdf(TScalarType x) const{
        return 0.5*(1+std::erf(x-mu)/(sigma*std::sqrt(2)));
    }

    TScalarType mean() const{
        return mu;
    }

    TScalarType variance() const{
        return sigma;
    }

    TScalarType mode() const{
        return mu;
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
    TScalarType operator()(TScalarType x) const{
        if(x<=0) throw std::string("InverseGaussianDensity: domain error. The inverse Gaussian density is not defined for x<=0.");

        return std::sqrt(lambda/(2*M_PI*x*x*x)) * std::exp(-lambda*(x-mu)*(x-mu)/(2*mu*mu*x));
    }


    TScalarType operator()() const{
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

    TScalarType cdf(TScalarType x) const{
        // gaussian cdf
//        auto g_cdf = [](TScalarType x, TScalarType mu, TScalarType sigma) -> TScalarType {
//            return  0.5*(1+std::erf((x-mu)/(sigma*std::sqrt(2))));
//        };

//        return g_cdf(std::sqrt(lambda/x)*(x/mu-1), mu, lambda) +
//                std::exp(2*lambda/mu) *
//                g_cdf(-std::sqrt(lambda/x)*(x/mu+1), mu, lambda);
//        double c=0;
//        for(double s=1e-10; s<x; s+=0.001){
//            c+=(*this)(s);
//        }
//        return c;
        GaussianDensity<TScalarType> g(0,1);
        return g.cdf(std::sqrt(lambda/x)*(x/mu-1)) +
                std::exp(2*lambda/mu) *
                g.cdf(-std::sqrt(lambda/x)*(x/mu+1));

    }

    TScalarType mean() const{
        return mu;
    }

    TScalarType variance() const{
        return mu*mu*mu/lambda;
    }

    TScalarType mode() const{
        return mu*(std::sqrt(1+9*mu*mu/(4*lambda*lambda)) - 3*mu/(2*lambda));
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
    TScalarType operator()(TScalarType x) const{
        if(x==0 && alpha < 1) throw std::string("GammaDensity: domain error. For alpha < 1 and x==0 the Gamma density is not defined.");
        if(x<0) throw std::string("GammaDensity: domain error. The Gamma density is not defined for x<0.");

        // compute log power
        std::complex<TScalarType> cx(x);
        std::complex<TScalarType> alphamin1(alpha-1);

        return std::abs(std::pow(cx,alphamin1)) / factor * std::exp(-x/beta);
    }

    TScalarType operator()() const{
        return distribution(Density<TScalarType>::g);
    }

    TScalarType cdf(TScalarType x) const{
        return 1/std::tgamma(alpha)*boost::math::tgamma_lower(alpha,beta*x);
    }

    TScalarType mean() const{
        return alpha/beta;
    }

    TScalarType variance() const{
        return alpha/(beta*beta);
    }

    TScalarType mode() const{
        return (alpha-1)/beta;
    }

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
    std::gamma_distribution<TScalarType> distribution;
};

}
