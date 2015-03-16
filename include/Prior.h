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
#include <functional>

#include <boost/math/special_functions/gamma.hpp>

namespace gpr {

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

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

    virtual std::string ToString() const = 0;
   
    // Bisection method to get the icdf
    // cdf(u) - u = 0
    // a has to be below cdf(u) and b has to be above cdf(u)
    TScalarType icdf(TScalarType u, TScalarType a=-1e8, TScalarType b=1e8) const{
        if(u<0 || u>1) throw std::string("Density::icdf: domain error. y is in [0,1]");
	if(sgn(this->cdf(a)-u) == sgn(this->cdf(b)-u)) throw std::string("Density::icdf: domain error. cdf(a)-u must have opposite sign than cdf(b)-u.");

	for(unsigned i=0; i<1000; i++){
	// Calculate c, the midpoint of the interval, c = 0.5 * (a + b)
	TScalarType c = 0.5 * (a+b);

	// Calculate the function value at the midpoint, f(c)
	TScalarType f = this->cdf(c)-u;

	// If convergence is satisfactory (that is, a - c is sufficiently small, 
	// or f(c) is sufficiently small), return c and stop iterating
	if(std::abs(a-c)<1e-10){ // || std::abs(f)<1e-14){
		return c;
	}
	
	// Examine the sign of f(c) and replace either (a, f(a)) or (b, f(b)) with (c, f(c)) 
	// so that there is a zero crossing within the new interval
	if(sgn(this->cdf(a)-u) != sgn(f)){
		b = c;
	}
	if(sgn(this->cdf(b)-u) != sgn(f)){
		a = c;
	}
	}

        throw std::string("Density::icdf: not converged after 1000 iterations.");
	return 0;
    }

protected:
    static std::default_random_engine g;

private:
    // reflection line: y = ax + c
    PairType reflect(TScalarType x, TScalarType y, TScalarType a, TScalarType c) const{
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
        return 1.0/(sigma*std::sqrt(2*M_PI)) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma));
    }

    TScalarType operator()() const{
        return normal_dist(Density<TScalarType>::g);
    }

    TScalarType GetDerivative(TScalarType x) const{
        return -(x-mu) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) / (std::sqrt(2)*std::sqrt(M_PI)*sigma*sigma*sigma);
    }

    TScalarType cdf(TScalarType x) const{
        return 0.5*(1+std::erf((x-mu)/(sigma*std::sqrt(2))));
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

    std::string ToString() const{
        return std::string("GaussianDensity");
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

    // Probability of x given lambda and mu
    TScalarType operator()(TScalarType x) const{
        if(x<=0) throw std::string("InverseGaussianDensity: domain error. The inverse Gaussian density is not defined for x<=0.");

        return std::sqrt(lambda/(2*M_PI*x*x*x)) * std::exp(-lambda*(x-mu)*(x-mu)/(2*mu*mu*x));
    }

    // Sample from probability density
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

    // log probability of x given lambda and mu
    TScalarType log(TScalarType x) const{
        if(x<0) throw std::string("InverseGaussianDensity::log: domain error. The log of the InverseGaussian is only defined for positive x.");
        TScalarType logx;
        if(x<std::numeric_limits<TScalarType>::epsilon()){
            logx = std::log(std::numeric_limits<TScalarType>::epsilon());
        }
        else{
            logx = std::log(x);
        }
        std::abs(1.0/x)*std::sqrt(lambda/(2*M_PI*logx*logx*logx)) * std::exp(-lambda*(logx-mu)*(logx-mu)/(2*mu*mu*logx));
    }


    // Probability of cumulative distribution
    TScalarType cdf(TScalarType x) const{
        if(x<=0) return 0;
        GaussianDensity<TScalarType> g(0,1);
        return g.cdf(std::sqrt(lambda/x)*(x/mu-1)) +
                std::min(std::exp(2*lambda/mu), std::numeric_limits<TScalarType>::max()) *
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

    std::string ToString() const{
        return std::string("InverseGaussianDensity");
    }

    static std::pair<TScalarType,TScalarType> GetMeanAndLambda(TScalarType mode, TScalarType variance){
        // since one has to solve a non-linear equation to get the mean
        // given the mode and variance, we perform Halley's method.
        //
        // Solve f(x) = 0
        //
        // xn+1 = xn - 2f(xn)f'(xn) / ( 2f'(xn)^2 - f(xn)*f''(xn) )
        //
        // as initial guess we use the value of the mode.


        // f: how to get mode given mean and variance (minus m to set f equal to zero)
        auto f = [](TScalarType mu, TScalarType m, TScalarType v)->TScalarType {
            return mu*std::sqrt(9*v*v/(4*mu*mu*mu*mu) +1) -3*v/(2*mu) - m;
        };

        // df: first derivative of f with respect to mu
        auto df = [](TScalarType mu, TScalarType m, TScalarType v)->TScalarType {
            return std::sqrt(9*v*v/(4*mu*mu*mu*mu) +1) - 9*v*v/(mu*mu*std::sqrt(9*v*v+4*mu*mu*mu*mu)) + 3*v/(2*mu*mu);
        };

        // ddf: second derivative of f with respect to mu
        auto ddf = [](TScalarType mu, TScalarType m, TScalarType v)->TScalarType {
            return -(3*v*(std::pow(9*v*v+4*mu*mu*mu*mu,3.0/2) - 27*v*v*v - 36*mu*mu*mu*mu*v))/(mu*mu*mu*std::pow(9*v*v+4*mu*mu*mu*mu,3.0/2));
        };

        typedef std::function<TScalarType(TScalarType,TScalarType,TScalarType)> Function;
        auto halley = [](TScalarType mu, TScalarType m, TScalarType v, Function f, Function df, Function ddf)->TScalarType {
            return mu - (2*f(mu,m,v)*df(mu,m,v))/(2*std::pow(df(mu,m,v),2.0)-f(mu,m,v)*ddf(mu,m,v));
        };

        // Halley's method
        TScalarType mu = mode*2; // initial value
        TScalarType mu_old = mu;
        for(unsigned i=0; i<100; i++){
            mu = halley(mu,mode,variance, f, df , ddf);
            if(std::abs(mu-mu_old)<1e-14){
                break;
            }
            mu_old = mu;
        }

        return std::make_pair(mu, mu*mu*mu/variance);
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

    std::string ToString() const{
        return std::string("GammaDensity");
    }

    static TScalarType GetAlpha(TScalarType mode, TScalarType variance){
        return (std::sqrt(mode*mode*(mode*mode+4*variance)) + mode*mode + 2*variance)/(2*variance);
    }
    static TScalarType GetBeta(TScalarType mode, TScalarType variance){
        return std::sqrt(GammaDensity<TScalarType>::GetAlpha(mode,variance)/variance);
    }

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
    std::gamma_distribution<TScalarType> distribution;
};

}
