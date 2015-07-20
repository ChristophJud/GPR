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
#include <iostream>
#include <memory>
#include <vector>

#include <boost/random.hpp>

#include <Eigen/Dense>

#include "Kernel.h"
#include "GaussianProcess.h"
#include "SparseGaussianProcess.h"

#include "Likelihood.h"
#include "SparseLikelihood.h"

namespace gpr{

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

template<class TScalarType>
class GaussianProcessInference{
public:
    typedef GaussianProcessInference Self;
    typedef std::shared_ptr<Self> Pointer;

    typedef GaussianProcess<TScalarType>                     GaussianProcessType;
    typedef typename GaussianProcessType::Pointer            GaussianProcessTypePointer;
    typedef typename GaussianProcessType::VectorListType     VectorListType;
    typedef typename GaussianProcessType::VectorType         VectorType;
    typedef typename GaussianProcessType::MatrixType         MatrixType;
    typedef std::vector<TScalarType>    ParameterVectorType;

    typedef Likelihood<TScalarType>             LikelihoodType;
    typedef typename LikelihoodType::Pointer    LikelihoodTypePointer;
    typedef typename LikelihoodType::ValueDerivativePair ValueDerivativePair;


    GaussianProcessInference(LikelihoodTypePointer lh, GaussianProcessTypePointer gp, double stepwidth, unsigned iterations) :
        m_Likelihood(lh),
        m_GaussianProcess(gp),
        m_StepWidth(stepwidth),
        m_StepWidth3(stepwidth*stepwidth*stepwidth),
        m_NumberOfIterations(iterations){

        m_Parameters = m_GaussianProcess->GetKernel()->GetParameters(); //.resize(m_GaussianProcess->GetKernel()->GetNumberOfParameters(), 1);
    }

    ~GaussianProcessInference(){}

    ParameterVectorType GetParameters(){
            return m_Parameters;
    }

    void Optimize(bool output=true, bool exp_output=false){

        for(unsigned i=0; i<m_NumberOfIterations; i++){
            // analytical
            try{

                m_GaussianProcess->GetKernel()->SetParameters(m_Parameters);

                ValueDerivativePair value_derivative = m_Likelihood->GetValueAndParameterDerivatives(m_GaussianProcess);
                VectorType likelihood_gradient = value_derivative.second;
                VectorType likelihood = value_derivative.first;

                VectorType parameter_update = (pinv<MatrixType>(likelihood_gradient * likelihood_gradient.adjoint()) * likelihood_gradient);
                if(output){

                    std::cout << "Likelihood " << value_derivative.first << ", : ";
                    for(unsigned p=0; p<m_Parameters.size(); p++){
                        if(!exp_output){
                            std::cout << m_Parameters[p] << ", ";
                        }
                        else{
                            std::cout << std::exp(m_Parameters[p]) << ", ";
                        }
                    }
                    std::cout << std::flush;

                    std::cout << ", Gradients: " << likelihood_gradient.adjoint() << std::flush;
                    std::cout << ", inf(J'J)J': " << parameter_update.adjoint() << std::flush;
                    std::cout << ", update: " << std::flush;
                }

                for(unsigned p=0; p<m_Parameters.size(); p++){
                    double u;
                    if(parameter_update[p]==0){ // log gradient step
                        if(likelihood_gradient[p]>=0){
                            u = m_StepWidth3*std::log(1+likelihood_gradient[p]);
                        }
                        else{
                            u = -m_StepWidth3*std::log(1+std::fabs(likelihood_gradient[p]));
                        }
                        m_Parameters[p] += u;
                    }
                    else{ // Gauss Newton step
                        u = parameter_update[p]*likelihood[0];
                        if(u>0){
                             u = m_StepWidth*std::log(1+u);
                        }
                        else{
                            u = -m_StepWidth*std::log(1+std::fabs(u));
                        }
                        m_Parameters[p] -= u;
                    }
                    if(output)std::cout << u << " " << std::flush;
                }

                if(output){
                    std::cout << ", new parameters: " << std::flush;
                    for(unsigned p=0; p<m_Parameters.size(); p++){
                        if(!exp_output) std::cout << m_Parameters[p] << ", " << std::flush;
                        else std::cout << std::exp(m_Parameters[p]) << ", " << std::flush;

                    }
                    std::cout << std::endl;
                }
            }
            catch(std::string& s){
                std::cout << "[failed] " << s << std::endl;
                return;
            }
        }
    }

private:
    TScalarType m_StepWidth;
    TScalarType m_StepWidth3;
    unsigned m_NumberOfIterations;

    LikelihoodTypePointer m_Likelihood;
    GaussianProcessTypePointer m_GaussianProcess;
    ParameterVectorType m_Parameters;

    GaussianProcessInference(const Self&); //purposely not implemented
    void operator=(const Self&); //purposely not implemented
};

} // namespace gpr
