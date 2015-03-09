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
 * Gamma Density: to evaluate the distribution at a point x
 *
 * p(x|alpha,beta) = 1/beta^alpha * x^{alpha-1} * exp(-x/alpha)
 *
 */
template<class TScalarType>
class GammaDensity : public Density<TScalarType>{
public:
    GammaDensity(TScalarType alpha, TScalarType beta) : alpha(alpha), beta(beta) {
        factor = std::pow(beta,alpha) * std::tgamma(alpha);
        distribution = std::gamma_distribution<TScalarType>(alpha, beta);
    }
    ~GammaDensity(){}
    TScalarType operator()(TScalarType x){
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
 * p(x|alpha,beta) = 1/beta^alpha * log(x^{alpha-1}) * exp(-log(x)/alpha)
 *
 */
template<class TScalarType>
class LogGammaDensity : public Density<TScalarType>{
public:
    LogGammaDensity(TScalarType alpha, TScalarType beta) : alpha(alpha), beta(beta) {
        factor = std::pow(beta,alpha) * std::tgamma(alpha);
    }
    ~LogGammaDensity(){}
    TScalarType operator()(TScalarType x){
        if(std::isnan(std::log(x)) || std::isinf(std::log(x))) throw std::string("LogGammaDensity: domain error.");

        // compute log power
        std::complex<TScalarType> logx(std::log(x));
        std::complex<TScalarType> alphamin1(alpha-1);

        return std::abs(std::pow(logx,alphamin1)) / factor * std::exp(-std::log(x)/beta);
    }

private:
    TScalarType alpha;
    TScalarType beta;
    TScalarType factor;
};

}
