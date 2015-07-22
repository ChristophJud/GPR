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

#include <Eigen/Dense>
#include <Eigen/SVD>

namespace gpr {

/**
 * An SVD based implementation of the Moore-Penrose pseudo-inverse
 */
template<class TMatrixType>
TMatrixType pinv(const TMatrixType& m, double epsilon = std::numeric_limits<double>::epsilon()) {
    typedef Eigen::JacobiSVD<TMatrixType> SVD;
    SVD svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
    typedef typename SVD::SingularValuesType SingularValuesType;
    const SingularValuesType singVals = svd.singularValues();
    SingularValuesType invSingVals = singVals;
    for(int i=0; i<singVals.rows(); i++) {
        if(singVals(i) <= epsilon) {
            invSingVals(i) = 0.0; // FIXED can not be safely inverted
        }
        else {
            invSingVals(i) = 1.0 / invSingVals(i);
        }
    }
    return TMatrixType(svd.matrixV() *
            invSingVals.asDiagonal() *
            svd.matrixU().transpose());
}


template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

// Superclass for density functions
// all densities have to be evaluatable
template<class TScalarType>
class Density{
public:

    typedef std::pair<TScalarType, TScalarType> PairType;
    typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;
    typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;

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

    TScalarType log(TScalarType x) const{
        TScalarType normalization = std::log(1.0/(sigma*std::sqrt(2*M_PI)));
        return normalization - (x-mu)*(x-mu)/(2*sigma*sigma);
    }

    TScalarType GetDerivative(TScalarType x) const{
        return -(x-mu) * std::exp(-(x-mu)*(x-mu)/(2*sigma*sigma)) / (std::sqrt(2)*std::sqrt(M_PI)*sigma*sigma*sigma);
    }

    TScalarType GetLogDerivative(TScalarType x) const{
        return -(x-mu)/(sigma*sigma);
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
 * Gaussian Density: to evaluate the distribution at a point x
 *
 * p(x|mu,sigma) =
 *
 */
template<class TScalarType>
class LogGaussianDensity : public Density<TScalarType>{
public:

    typedef Density<TScalarType> Superclass;
    typedef LogGaussianDensity<TScalarType> Self;
    typedef typename Superclass::VectorType VectorType;
    typedef typename Superclass::MatrixType MatrixType;

    typedef typename std::pair<TScalarType,TScalarType> ParameterPairType;

    LogGaussianDensity (TScalarType mu, TScalarType sigma) : m_mu(mu), m_sigma(sigma) {
        if(m_sigma<=0) throw std::string("LogGaussianDensity : the LogGaussian density is only defined for sigma>0");
        standard_normal = std::normal_distribution<TScalarType>(0, 1);
    }
    ~LogGaussianDensity (){}
    TScalarType operator()(TScalarType x) const{
        if(x<=0) throw std::string("LogGaussianDensity : domain error. x has to be greater than zero");
        return 1.0/(x*m_sigma*std::sqrt(2*M_PI)) * std::exp(-(std::log(x)-m_mu)*(std::log(x)-m_mu)/(2*m_sigma*m_sigma));
    }

    TScalarType operator()() const{
        return std::exp(m_mu + m_sigma * standard_normal(Density<TScalarType>::g));
    }

    TScalarType log(TScalarType x) const{
        return std::log(this->operator()(x));
    }

    TScalarType GetDerivative(TScalarType x) const{
        if(x<=0) throw std::string("LogGaussianDensity : domain error. x has to be greater than zero");
        TScalarType logx = std::log(x);
        TScalarType f = std::exp(-(logx*logx-2*m_mu*logx+m_mu*m_mu)/(2*m_sigma*m_sigma));
        return -(f*(logx+m_sigma*m_sigma-m_mu))/(std::sqrt(2)*std::sqrt(M_PI)*m_sigma*m_sigma*m_sigma*x*x);
    }

    TScalarType GetLogDerivative(TScalarType x) const{
        if(x<=0) throw std::string("LogGaussianDensity : domain error. x has to be greater than zero");
        return - (std::log(x) + m_sigma*m_sigma - m_mu)/(m_sigma*m_sigma*x);
    }

    TScalarType cdf(TScalarType x) const{
        if(x<=0) throw std::string("LogGaussianDensity : domain error. x has to be greater than zero");
        return 0.5 + 0.5*std::erf((std::log(x)-m_mu)/(std::sqrt(2)*m_sigma));
    }

    TScalarType mean() const{
        return std::exp(m_mu+m_sigma*m_sigma/2);
    }

    TScalarType variance() const{
        return (std::exp(m_sigma*m_sigma)-1)*std::exp(2*m_mu+m_sigma*m_sigma);
    }

    TScalarType mode() const{
        return std::exp(m_mu-m_sigma*m_sigma);
    }

    std::string ToString() const{
        return std::string("LogGaussianDensity");
    }

    // Attention: convergence is not stable for peaked mode
    static ParameterPairType GetMuAndSigma(TScalarType mode, TScalarType variance){
        // since one has to solve a non-linear equation to get the mean
        // given the mode and variance, we perform Halley's method.
        //
        // Solve f(x) = 0
        //
        // xn+1 = xn - 2f(xn)f'(xn) / ( 2f'(xn)^2 - f(xn)*f''(xn) )
        //
        // as initial guess we use the value of the mode.

        auto f1 = [](long double mu, long double s, long double m) -> long double {
            return std::exp(mu-s*s) - m;
        };

        auto df1dmu = [](long double mu, long double s) -> long double {
            return std::exp(mu-s*s);
        };

        auto df1ds = [](long double mu, long double s) -> long double {
            return -2*s*std::exp(mu-s*s);
        };

        auto ddf1dds = [](long double mu, long double s) -> long double {
            return 2*(2*s*s-1)*std::exp(mu-s*s);
        };

        auto f2 = [](long double mu, long double s, long double v) -> long double {
            return (std::exp(s*s)-1)*std::exp(2*mu+s*s) - v;
        };

        auto df2dmu = [](long double mu, long double s) -> long double {
            return 2*(std::exp(s*s)-1)*std::exp(2*mu+s*s);
        };

        auto df2ds = [](long double mu, long double s) -> long double {
            return 2*s*(2*std::exp(s*s)-1)*std::exp(2*mu+s*s);
        };

        auto ddf2ddmu = [](long double mu, long double s) -> long double {
            return 4*(std::exp(s*s)-1)*std::exp(2*mu+s*s);
        };

        auto ddf2dds = [](long double mu, long double s) -> long double {
            return 2*((8*s*s+2)*std::exp(s*s)-2*s*s-1)*std::exp(s*s+2*mu);
        };

        auto ddf2dmuds = [](long double mu, long double s) -> long double {
            return 4*s*(2*std::exp(s*s)-1)*std::exp(s*s+2*mu);
        };


        auto halley = [&f1, &f2, &df1dmu, &df1ds, &df2dmu, &df2ds, &ddf1dds, &ddf2dds, &ddf2ddmu, &ddf2dmuds]
                (long double mu, long double s, long double m, long double v)->VectorType {

            typedef Eigen::Matrix<long double, Eigen::Dynamic, 1> VectorType;
            typedef Eigen::Matrix<long double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;

            // a = 0.5 -> Halley's method
            // a = -inf -> Newton method
            TScalarType alpha = 0.5;

            VectorType F = VectorType::Zero(2);
            F(0) = f1(mu,s,m);
            F(1) = f2(mu,s,v);

            MatrixType J = MatrixType::Zero(2,2);
            J(0,0) = df1dmu(mu,s);
            J(0,1) = df2dmu(mu,s);
            J(1,0) = df1ds(mu,s);
            J(1,1) = df2ds(mu,s);

            MatrixType H = MatrixType::Zero(2,4);
            H(0,0) = J(0,0);
            H(0,1) = J(0,1);
            H(0,2) = ddf2ddmu(mu,s);
            H(0,3) = ddf2dmuds(mu,s);
            H(1,0) = H(0,1);
            H(1,1) = ddf1dds(mu,s);
            H(1,2) = H(0,3);
            H(1,3) = ddf2dds(mu,s);

            VectorType nc = -pinv<MatrixType>(J)*F; // Newton correction
            MatrixType A = MatrixType::Zero(2,2);
            A.row(0) = (H.leftCols(2)*nc).adjoint();
            A.row(1) = (H.rightCols(2)*nc).adjoint();

            VectorType convexity = -0.5*pinv<MatrixType>(J+alpha*A)*A*nc;
            VectorType hc = nc+convexity; // Halley correction

            VectorType u = VectorType::Zero(2);
            u[0] = mu+hc[0];
            u[1] = s+hc[1];

            return u.cast<TScalarType>();
        };

        // Halley's method

        bool cout = false;

        // initial guess of mu=p(0) and sigma=p(1)
        // iterative
        VectorType p = VectorType::Zero(2);
        p(1) = 0; // initial value

        TScalarType average = 0;
        unsigned cnt = 0;
        for(unsigned i=0; i<20; i++){
            if(cout) std::cout << p(1) << std::endl;
            p(1) = std::sqrt(std::log(1+variance/std::exp(std::log(mode)+3/2.0*p(1)*p(1))));

            if(i>10){
                average += p(1);
                cnt++;
            }
        }
        if(cnt>0){
            p(1) = average/cnt;
        }
        if(cout) std::cout << "final sigma " << p(1) << std::endl;

        TScalarType mean = std::exp(std::log(mode)+3/2.0*p(1)*p(1));

        if(cout) std::cout << "mean " << mean << std::endl;

        //p(0) = std::log(mean/std::sqrt(1+variance/(mean*mean)));
        p(0) = std::log(mode)+p(1)*p(1);

        if(cout){
            std::cout << "mu " << p(0) << std::endl;
            std::cout << "initial guess: mu/sigma " << p.adjoint() << std::endl;

            Self density(p(0), p(1));
            std::cout << "mu/sigma: " << p.adjoint() << ", \t mode/variance: " << density.mode() << " " << density.variance() << std::endl;
        }
        VectorType p_old = p;
        for(unsigned i=0; i<100; i++){
            p = halley(p(0), p(1), mode, variance);
            if(cout){
                Self density(p(0), p(1));
                std::cout << "mu/sigma: " << p.adjoint() << ", \t mode/variance: " << density.mode() << " " << density.variance() << std::endl;
            }
            if((p-p_old).norm()<1e-14){
                break;
            }
            p_old = p;
        }

        TScalarType mu = p(0);
        TScalarType s = p(1);
        TScalarType err_mode = std::fabs(std::exp(mu-s*s) - mode);
        TScalarType err_variance = std::fabs((std::exp(s*s)-1)*std::exp(2*mu+s*s)-variance);
        if(err_mode > 1e-10 || err_variance > 1e-10){
            std::stringstream ss;
            ss << "LogGaussianDensity::GetMuAndSigma: cannot determ mu and sigma for mode=" << mode << " and variance=" << variance;
            ss << ". Errors: mode " << err_mode << ", variance " << err_variance;
            throw ss.str();
        }

        if(std::isnan(mu) || std::isnan(s)){
            throw std::string("LogGaussianDensity::GetMuAndSigma: result is nan");
        }
        return std::make_pair(mu, std::fabs(s));
    }


private:
    TScalarType m_mu;
    TScalarType m_sigma;
    std::normal_distribution<TScalarType> standard_normal;
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
    typedef Density<TScalarType> Superclass;
    typedef InverseGaussianDensity<TScalarType> Self;
    typedef std::pair<TScalarType,TScalarType> ParameterPairType;

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

    TScalarType GetDerivative(TScalarType x) const{
        TScalarType numerator = -lambda*(lambda*x*x+3*mu*mu*x-lambda*mu*mu)*std::exp(-(lambda*x*x-2*lambda*mu*x+lambda*mu*mu)/(2*mu*mu*x));
        TScalarType denumerator = std::sqrt(6)*std::sqrt(M_PI)*mu*mu*std::sqrt(lambda/(x*x*x))*x*x*x*x*x;
        return numerator/denumerator;
    }

    TScalarType GetLogDerivative(TScalarType x) const{
        return -3/(2*x)+lambda/(2*x*x)-lambda/(2*mu*mu);
    }

    // log probability of x given lambda and mu
    TScalarType logDensity(TScalarType x) const{
        if(x<=1) throw std::string("InverseGaussianDensity::log: domain error. The log of the InverseGaussian is only defined for x>1.");
        TScalarType logx;
        if(x<std::numeric_limits<TScalarType>::epsilon()){
            logx = std::log(std::numeric_limits<TScalarType>::epsilon());
        }
        else{
            logx = std::log(x);
        }

        TScalarType root_value;
        if(x==1){
            root_value = std::sqrt(std::numeric_limits<TScalarType>::max());
        }
        else{
            std::complex<TScalarType> f(lambda/(2*M_PI*logx*logx*logx));
            root_value = std::abs(std::sqrt(f));
        }

        TScalarType exp_value = std::exp(-lambda*(logx-mu)*(logx-mu)/(2*mu*mu*logx));
        if(exp_value==0) exp_value = std::numeric_limits<TScalarType>::epsilon();

        //std::cout << x << ", " << std::abs(1.0/x) << ", " << root_value << ", " << exp_value << ", " << std::abs(1.0/x) * root_value * exp_value << std::endl;



        return std::abs(1.0/x) * root_value * exp_value;
    }

    TScalarType log(TScalarType x) const{
        return std::log(this->operator()(x));
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

    static ParameterPairType GetMeanAndLambda(TScalarType mode, TScalarType variance){
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
        std::cout << "iteration" << std::endl;
        for(unsigned i=0; i<100; i++){
            std::cout << mu << std::endl;
            mu = halley(mu,mode,variance, f, df , ddf);
            if(std::abs(mu-mu_old)<1e-14){
                break;
            }
            mu_old = mu;
        }

        Self p(mu, mu*mu*mu/variance);
        std::cout << "mode " << mode << ", estimation " << p.mode() << std::endl;
        std::cout << "variance " << variance << ", estimation " << p.variance() << std::endl;
        if(std::fabs(p.mode() - mode)>1e-14){
            std::stringstream ss;
            ss << "InverseGaussianDensity::GetMeanAndLambda: cannot determ mean and lambda for mode=" << mode << " and variance=" << variance;
            throw ss.str();
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

        return std::pow(beta,alpha)/std::tgamma(alpha) * std::abs(std::pow(cx,alphamin1)) * std::exp(-x/beta);
    }

    TScalarType log(TScalarType x) const{
        if(x<=0) throw std::string("GammaDensity::log: domain error. The log of Gamma is only defined for x>0.");
        TScalarType logx;
        if(x<std::numeric_limits<TScalarType>::epsilon()){
            logx = std::log(std::numeric_limits<TScalarType>::epsilon());
        }
        else{
            logx = std::log(x);
        }

        // compute log power
        std::complex<TScalarType> cx(x);
        std::complex<TScalarType> clogx(logx);
        std::complex<TScalarType> alphamin1(alpha-1);

        return std::pow(beta,alpha)/(std::abs(x)*std::tgamma(alpha)) * std::abs(std::pow(clogx,alphamin1)) * std::exp(-logx/beta);
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
