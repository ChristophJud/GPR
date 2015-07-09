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

#include <fstream>
#include <iostream>
#include <iomanip>

#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "GaussianProcess.h"
#include "KernelFactory.h"
#include "MatrixIO.h"
#include "LAPACKUtils.h"

namespace gpr{

template< class TScalarType >
void GaussianProcess<TScalarType>::AddSample(const typename GaussianProcess<TScalarType>::VectorType &x,
                                             const typename GaussianProcess<TScalarType>::VectorType &y){
    if(m_SampleVectors.size() == 0){ // first call of AddSample defines dimensionality of input space
        m_InputDimension = x.size();
    }
    if(m_LabelVectors.size() == 0){ // first call of AddSample defines dimensionality of output space
        m_OutputDimension = y.size();
    }

    CheckInputDimension(x, "GaussianProcess::AddSample: ");
    CheckOutputDimension(y, "GaussianProcess::AddSample: ");

    m_SampleVectors.push_back(x);
    m_LabelVectors.push_back(y);
    m_Initialized = false;
}

template< class TScalarType >
typename GaussianProcess<TScalarType>::VectorType
GaussianProcess<TScalarType>::Predict(const typename GaussianProcess<TScalarType>::VectorType &x){
    Initialize();
    CheckInputDimension(x, "GaussianProcess::Predict: ");
    VectorType Kx;
    ComputeKernelVector(x, Kx);
    return (Kx.adjoint() * m_RegressionVectors).adjoint();
}

template< class TScalarType >
typename GaussianProcess<TScalarType>::VectorType
GaussianProcess<TScalarType>::PredictDerivative(const typename GaussianProcess<TScalarType>::VectorType &x,
                                                           typename GaussianProcess<TScalarType>::MatrixType &D){
    Initialize();
    CheckInputDimension(x, "GaussianProcess::PredictDerivative: ");
    VectorType Kx;
    ComputeKernelVector(x, Kx);
    MatrixType X;
    ComputeDifferenceMatrix(x, X);

    unsigned d = m_InputDimension;
    unsigned m = m_OutputDimension;
    D.resize(m_InputDimension, m_OutputDimension);
    for(unsigned i=0; i<m_OutputDimension; i++){
        D.col(i) = -X.transpose() * Kx.cwiseProduct(m_RegressionVectors.col(i));
    }
    return (Kx.adjoint() * m_RegressionVectors).adjoint(); // return point prediction
}

template< class TScalarType >
TScalarType
GaussianProcess<TScalarType>::operator()(const typename GaussianProcess<TScalarType>::VectorType & x,
                                         const typename GaussianProcess<TScalarType>::VectorType & y){
    Initialize();
    CheckInputDimension(x, "GaussianProcess::(): ");
    CheckInputDimension(y, "GaussianProcess::(): ");
    VectorType Kx;
    ComputeKernelVector(x, Kx);
    VectorType Ky;
    ComputeKernelVector(y, Ky);

    if(m_CoreMatrix.diagonalSize() == 0){
        ComputeCoreMatrix(m_CoreMatrix);
    }
    return (*m_Kernel)(x, y) - Kx.adjoint() * m_CoreMatrix * Ky;
}

template< class TScalarType >
TScalarType
GaussianProcess<TScalarType>::GetCredibleInterval(const typename GaussianProcess<TScalarType>::VectorType& x){
    Initialize();
    CheckInputDimension(x, "GaussianProcess::GetCredibleIntervall: ");

    // due to nummerical instabilities of the inversion of the kernel matrix
    // gp(x,x) might return negative values
    // therefore the maximum of zero and c is taken.
    TScalarType c = (*this)(x, x);
    if(debug && c<0) std::cout << "GaussianProcess::GetCredibleIntervall: prediction is instable. gp(x,x) = " << c << "." << std::endl;
    c = 2*std::sqrt(std::max(static_cast<TScalarType>(0.0),c));
    return c;
}


template< class TScalarType >
void GaussianProcess<TScalarType>::Initialize(){
    if(m_Initialized){
        return;
    }
    if(!(m_SampleVectors.size() > 0)){
        throw std::string("GaussianProcess::Initialize: no input samples defined during initialization");
    }
    if(!(m_LabelVectors.size() > 0)){
        throw std::string("GaussianProcess::Initialize: no ouput labels defined during initialization");
    }
    ComputeRegressionVectors();
    m_Initialized = true;
}


template< class TScalarType >
void GaussianProcess<TScalarType>::Save(std::string prefix){
    if(!m_Initialized){
        throw std::string("GaussianProcess::Save: gaussian process is not initialized.");
    }

    if(debug){
        std::cout << "GaussianProcess::Save: writing gaussian process: " << std::endl;
        std::cout << "\t " << prefix+"-RegressionVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-CoreMatrix.txt" << std::endl;
        std::cout << "\t " << prefix+"-SampleVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-LabelVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-ParameterFile.txt" << std::endl;
    }

    // save regression vectors
    WriteMatrix<MatrixType>(m_RegressionVectors, prefix+"-RegressionVectors.txt");

    // save regression vectors
    if(m_EfficientStorage) m_CoreMatrix.setZero(0,0);
    WriteMatrix<MatrixType>(m_CoreMatrix, prefix+"-CoreMatrix.txt");

    // save sample vectors
    MatrixType X = MatrixType::Zero(m_SampleVectors[0].size(), m_SampleVectors.size());
    for(unsigned i=0; i<m_SampleVectors.size(); i++){
        X.block(0,i,m_SampleVectors[0].size(),1) = m_SampleVectors[i];
    }
    WriteMatrix<MatrixType>(X, prefix+"-SampleVectors.txt");

    // save label vectors
    MatrixType Y = MatrixType::Zero(m_LabelVectors[0].size(), m_LabelVectors.size());
    for(unsigned i=0; i<m_LabelVectors.size(); i++){
        Y.block(0,i,m_LabelVectors[0].size(),1) = m_LabelVectors[i];
    }
    WriteMatrix<MatrixType>(Y, prefix+"-LabelVectors.txt");

    // save parameters
    // KernelType, #KernelParameters, KernelParameters, noise, InputDimension, OutputDimension
    std::ofstream parameter_outfile;
    parameter_outfile.open(std::string(prefix+"-ParameterFile.txt").c_str());

    parameter_outfile << m_Sigma << " " << m_InputDimension << " " << m_OutputDimension << " " << m_EfficientStorage << " " << debug << " ";

    parameter_outfile << std::setprecision(std::numeric_limits<TScalarType>::digits10 +1);
    parameter_outfile << m_Kernel->ToString();

    parameter_outfile.close();
}


template< class TScalarType >
void GaussianProcess<TScalarType>::Load(std::string prefix){
    if(debug){
        std::cout << "GaussianProcess::Load: loading gaussian process: " << std::endl;
        std::cout << "\t " << prefix+"-RegressionVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-CoreMatrix.txt" << std::endl;
        std::cout << "\t " << prefix+"-SampleVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-LabelVectors.txt" << std::endl;
        std::cout << "\t " << prefix+"-ParameterFile.txt" << std::endl;
    }

    // load regression vectors
    std::string rv_filename = prefix+"-RegressionVectors.txt";
    fs::path rv_file(rv_filename.c_str());
    if(!(fs::exists(rv_file) && !fs::is_directory(rv_file))){
        throw std::string("GaussianProcess::Load: "+rv_filename+" does not exist or is a directory.");
    }
    m_RegressionVectors = ReadMatrix<MatrixType>(rv_filename);

    // load core matrix
    std::string cm_filename = prefix+"-CoreMatrix.txt";
    fs::path cm_file(cm_filename.c_str());
    if(!(fs::exists(cm_file) && !fs::is_directory(cm_file))){
        throw std::string("GaussianProcess::Load: "+cm_filename+" does not exist or is a directory.");
    }
    m_CoreMatrix = ReadMatrix<MatrixType>(cm_filename);

    // load sample vectors
    std::string sv_filename = prefix+"-SampleVectors.txt";
    fs::path sv_file(sv_filename.c_str());
    if(!(fs::exists(sv_file) && !fs::is_directory(sv_file))){
        throw std::string("GaussianProcess::Load: "+sv_filename+" does not exist or is a directory.");
    }
    MatrixType X = ReadMatrix<MatrixType>(sv_filename);
    m_SampleVectors.clear();
    for(unsigned i=0; i<X.cols(); i++){
        m_SampleVectors.push_back(X.col(i));
    }

    // load label vectors
    std::string lv_filename = prefix+"-LabelVectors.txt";
    fs::path lv_file(lv_filename.c_str());
    if(!(fs::exists(lv_file) && !fs::is_directory(lv_file))){
        throw std::string("GaussianProcess::Load: "+lv_filename+" does not exist or is a directory.");
    }
    MatrixType Y = ReadMatrix<MatrixType>(lv_filename);
    m_LabelVectors.clear();
    for(unsigned i=0; i<Y.cols(); i++){
        m_LabelVectors.push_back(Y.col(i));
    }

    // load parameters
    std::string pf_filename = prefix+"-ParameterFile.txt";
    if(!(fs::exists(pf_filename) && !fs::is_directory(pf_filename))){
        throw std::string("GaussianProcess::Load: "+pf_filename+" does not exist or is a directory.");
    }

    std::ifstream parameter_infile;
    parameter_infile.open( pf_filename.c_str() );

    // reading parameter file
    std::string line;
    bool load = true;
    if(std::getline(parameter_infile, line)) {
        std::stringstream line_stream(line);

        if(!(line_stream >> m_Sigma &&
             line_stream >> m_InputDimension &&
             line_stream >> m_OutputDimension &&
             line_stream >> m_EfficientStorage &&
             line_stream >> debug) ||
                load == false){
            throw std::string("GaussianProcess::Load: parameter file is corrupt");
        }

        std::string kernel_string;
        line_stream >> kernel_string;

        m_Kernel = KernelFactory<TScalarType>::GetKernel(kernel_string);

    }
    parameter_infile.close();


    m_Initialized = true;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ToString() const{
    std::cout << "---------------------------------------" << std::endl;
    std::cout << "Gaussian Process" << std::endl;
    std::cout << " - initialized:\t\t" << m_Initialized << std::endl;
    std::cout << " - # samples:\t\t" << m_SampleVectors.size() << std::endl;
    std::cout << " - # labels:\t\t" << m_LabelVectors.size() << std::endl;
    std::cout << " - noise:\t\t" << m_Sigma << std::endl;
    std::cout << " - input dimension:\t" << m_InputDimension << std::endl;
    std::cout << " - output dimension:\t" << m_OutputDimension << std::endl;
    std::cout << std::endl;
    std::cout << " - Kernel:" << std::endl;
    std::cout << "       - Type:\t\t" << m_Kernel->ToString() << std::endl;
    std::cout << "       - Parameter:\t";
    for(unsigned i=0; i<m_Kernel->GetParameters().size(); i++){
        std::cout << m_Kernel->GetParameters()[i] << ", ";
    }
    std::cout << std::endl;
    std::cout << "---------------------------------------" << std::endl;
}

template< class TScalarType >
bool GaussianProcess<TScalarType>::operator ==(const GaussianProcess<TScalarType> &b) const{
    if(this->debug) std::cout << "GaussianProcess::comparison: " << std::flush;

    if((this->m_RegressionVectors - b.m_RegressionVectors).norm() > 0){
        if(this->debug) std::cout << "regression vectors not equal." << std::endl;
        return false;
    }

    if(this->m_CoreMatrix.diagonalSize() != b.m_CoreMatrix.diagonalSize()){
        if(this->debug) std::cout << "core matrices not equal."  << std::endl;
        return false;
    }
    else{
        if(this->debug && (this->m_CoreMatrix - b.m_CoreMatrix).norm() > 0) std::cout << "core matrices error is " << (this->m_CoreMatrix - b.m_CoreMatrix).norm() << std::endl;
    }


    if(this->m_SampleVectors.size() != b.m_SampleVectors.size()){
        if(this->debug) std::cout << "number of sample vectors not equal." << std::endl;
        return false;
    }
    for(unsigned i=0; i<this->m_SampleVectors.size(); i++){
        if((this->m_SampleVectors[i] - b.m_SampleVectors[i]).norm()>0){
            if(this->debug) std::cout << "sample vectors not equal." << std::endl;
            return false;
        }
    }

    if(this->m_LabelVectors.size() != b.m_LabelVectors.size()){
        if(this->debug) std::cout << "number of label vectors not equal." << std::endl;
        return false;
    }
    for(unsigned i=0; i<this->m_LabelVectors.size(); i++){
        if((this->m_LabelVectors[i] - b.m_LabelVectors[i]).norm()>0) {
            if(this->debug) std::cout << "label vectors not equal." << std::endl;
            return false;
        }
    }
    if(*this->m_Kernel.get() != *b.m_Kernel.get()){
        if(this->debug) std::cout << "kernel not equal." << std::endl;
        return false;
    }
    if(this->m_Sigma != b.m_Sigma){
        if(this->debug) std::cout << "sigma not equal." << std::endl;
        return false;
    }
    if(this->m_Initialized != b.m_Initialized){
        if(this->debug) std::cout << "initialization state not equal." << std::endl;
        return false;
    }
    if(this->m_InputDimension != b.m_InputDimension){
        if(this->debug) std::cout << "input dimension not equal." << std::endl;
        return false;
    }
    if(this->m_OutputDimension != b.m_OutputDimension){
        if(this->debug) std::cout << "output dimension not equal." << std::endl;
        return false;
    }
    if(this->m_EfficientStorage!= b.m_EfficientStorage){
        if(this->debug) std::cout << "efficient storage setting not equal." << std::endl;
        return false;
    }
    if(this->debug != b.debug){
        if(this->debug) std::cout << "debug state not equal." << std::endl;
        return false;
    }
    if(this->debug) std::cout << "is equal!" << std::endl;
    return true;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeKernelMatrix(typename GaussianProcess<TScalarType>::MatrixType &M) const{
    if(debug){
        std::cout << "GaussianProcess::ComputeKernelMatrix: building kernel matrix... ";
        std::cout.flush();
    }

    ComputeKernelMatrixInternal(M, m_SampleVectors);

    if(debug) std::cout << "[done]" << std::endl;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::AddNoiseToKernelMatrix(typename GaussianProcess<TScalarType>::MatrixType &M) const{
    // add noise variance to diagonal
    for(unsigned i=0; i<M.rows(); i++){
        M(i,i) += m_Sigma;
    }
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeKernelMatrixInternal(typename GaussianProcess<TScalarType>::MatrixType &M,
                                                               const typename GaussianProcess<TScalarType>::VectorListType& samples) const{
    unsigned n = samples.size();
    M.resize(n,n);

#pragma omp parallel for
    for(unsigned i=0; i<n; i++){
        for(unsigned j=i; j<n; j++){
            TScalarType v = (*m_Kernel)(samples[i],samples[j]);
            M(i,j) = v;
            M(j,i) = v;
        }
    }
}


template< class TScalarType >
TScalarType GaussianProcess<TScalarType>::ComputeKernelMatrixTrace() const{
    if(debug){
        std::cout << "GaussianProcess::ComputeKernelMatrixTrace: sum up diagonal elements of kernel matrix... ";
        std::cout.flush();
    }

    TScalarType trace = ComputeKernelMatrixTraceInternal(m_SampleVectors);

    if(debug) std::cout << "[done]" << std::endl;

    return trace;
}

template< class TScalarType >
TScalarType GaussianProcess<TScalarType>::ComputeKernelMatrixTraceInternal(const typename GaussianProcess<TScalarType>::VectorListType& samples) const{
    unsigned n = samples.size();
    TScalarType trace = 0;

    for(unsigned i=0; i<n; i++){
            trace += (*m_Kernel)(samples[i],samples[i]);
    }
    return trace;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeDerivativeKernelMatrix(typename GaussianProcess<TScalarType>::MatrixType &M) const{
    if(debug){
        std::cout << "GaussianProcess::ComputeDerivativeKernelMatrix: building kernel matrix... ";
        std::cout.flush();
    }

    unsigned num_params = m_Kernel->GetNumberOfParameters();

    unsigned n = m_SampleVectors.size();
    M.resize(n*num_params,n);

#pragma omp parallel for
    for(unsigned i=0; i<n; i++){
        for(unsigned j=i; j<n; j++){
            typename GaussianProcess<TScalarType>::VectorType v;
            v = m_Kernel->GetDerivative(m_SampleVectors[i], m_SampleVectors[j]);

            if(v.rows() != num_params) throw std::string("GaussianProcess::ComputeDerivativeKernelMatrix: dimension missmatch in derivative.");
            for(unsigned p=0; p<num_params; p++){

                //if(i+p*n >= M.rows() || j+p*n >= M.rows())  throw std::string("GaussianProcess::ComputeDerivativeKernelMatrix: dimension missmatch in derivative.");

                M(i + p*n, j) = v[p];
                M(j + p*n, i) = v[p];

                //if(i + p*n == j) M(i + p*n, j) += m_Sigma; // TODO: not sure if this is needed
            }
        }
    }
    //std::cout << "M: " << std::endl;
    //std::cout << M << std::endl;
    if(debug) std::cout << "[done]" << std::endl;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeCoreMatrix(typename GaussianProcess<TScalarType>::MatrixType &C) const{
    MatrixType K;
    ComputeKernelMatrix(K);

    // add noise variance to diagonal
    AddNoiseToKernelMatrix(K);


    C = InvertKernelMatrix(K, m_InvMethod);

    if(debug){
        std::cout << "GaussianProcess::ComputeCoreMatrix: inversion error: " << (K*C - MatrixType::Identity(K.rows(),K.cols())).norm() << std::endl;
    }
}

template< class TScalarType >
TScalarType GaussianProcess<TScalarType>::ComputeCoreMatrixWithDeterminant(typename GaussianProcess<TScalarType>::MatrixType &C) const{
    MatrixType K;
    ComputeKernelMatrix(K);

    // add noise variance to diagonal
    AddNoiseToKernelMatrix(K);

    C = InvertKernelMatrix(K, m_InvMethod);

    if(debug){
        std::cout << "GaussianProcess::ComputeCoreMatrix: inversion error: " << (K*C - MatrixType::Identity(K.rows(),K.cols())).norm() << std::endl;
        std::cout << "GaussianProcess::ComputeCoreMatrix: determinant of K: " << K.determinant() << std::endl;
    }
    return K.determinant();
}

template< class TScalarType >
typename GaussianProcess<TScalarType>::MatrixType GaussianProcess<TScalarType>::InvertKernelMatrix(const typename GaussianProcess<TScalarType>::MatrixType &K,
                                                      typename GaussianProcess<TScalarType>::InversionMethod inv_method = GaussianProcess<TScalarType>::FullPivotLU,
                                                                                                   bool stable) const{
    // compute core matrix
    if(debug){
        std::cout << "GaussianProcess::InvertKernelMatrix: inverting kernel matrix... ";
        std::cout.flush();
    }

    typename GaussianProcess<TScalarType>::MatrixType core;

    switch(inv_method){
    // standard method: fast but not that accurate
    // Uses the LU decomposition with full pivoting for the inversion
    case FullPivotLU:{
        if(debug) std::cout << " (inversion method: FullPivotLU) " << std::flush;
        try{
            if(stable){
                core = K.inverse();
            }
            else{
                if(debug) std::cout << " (using lapack) " << std::flush;
                core = lapack::lu_invert<TScalarType>(K);
            }
        }
        catch(lapack::LAPACKException& e){
            core = K.inverse();
        }
    }
    break;

    // very accurate and very slow method, use it for small problems
    // Uses the two-sided Jacobi SVD decomposition
    case JacobiSVD:{
        if(debug) std::cout << " (inversion method: JacobiSVD) " << std::flush;
        Eigen::JacobiSVD<MatrixType> jacobisvd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if((jacobisvd.singularValues().real().array() < 0).any() && debug){
            std::cout << "GaussianProcess::InvertKernelMatrix: warning: there are negative eigenvalues.";
            std::cout.flush();
        }
        core = jacobisvd.matrixV() * VectorType(1/jacobisvd.singularValues().array()).asDiagonal() * jacobisvd.matrixU().transpose();
    }
    break;

    // accurate method and faster than Jacobi SVD.
    // Uses the bidiagonal divide and conquer SVD
    case BDCSVD:{
        if(debug) std::cout << " (inversion method: BDCSVD) " << std::flush;
#ifdef EIGEN_BDCSVD_H
        Eigen::BDCSVD<MatrixType> bdcsvd(K, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if((bdcsvd.singularValues().real().array() < 0).any() && debug){
            std::cout << "GaussianProcess::InvertKernelMatrix: warning: there are negative eigenvalues.";
            std::cout.flush();
        }
        core = bdcsvd.matrixV() * VectorType(1/bdcsvd.singularValues().array()).asDiagonal() * bdcsvd.matrixU().transpose();
#else
        // this is checked, since BDCSVD is currently not in the newest release
        throw std::string("GaussianProcess::InvertKernelMatrix: BDCSVD is not supported by the provided Eigen library.");
#endif

    }
    break;

    // faster than the SVD method but less stable
    // computes the eigenvalues/eigenvectors of selfadjoint matrices
    case SelfAdjointEigenSolver:{
        if(debug) std::cout << " (inversion method: SelfAdjointEigenSolver) " << std::flush;
        try{
            core = lapack::chol_invert<TScalarType>(K);
        }
        catch(lapack::LAPACKException& e){
            Eigen::SelfAdjointEigenSolver<MatrixType> es;
            es.compute(K);
            VectorType eigenValues = es.eigenvalues().reverse();
            MatrixType eigenVectors = es.eigenvectors().rowwise().reverse();
            if((eigenValues.real().array() < 0).any() && debug){
                std::cout << "GaussianProcess::InvertKernelMatrix: warning: there are negative eigenvalues.";
                std::cout.flush();
            }
            core = eigenVectors * VectorType(1/eigenValues.array()).asDiagonal() * eigenVectors.transpose();
        }
    }
    break;
    }

    if(debug) std::cout << "[done]" << std::endl;
    return core;
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeLabelMatrix(typename GaussianProcess<TScalarType>::MatrixType &Y) const{
    ComputeLabelMatrixInternal(Y, m_LabelVectors);
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeLabelMatrixInternal(typename GaussianProcess<TScalarType>::MatrixType &Y,
                                                              const typename GaussianProcess<TScalarType>::VectorListType& labels) const{
    unsigned n = labels.size();
    if(!(n > 0)){
        throw std::string("GaussianProcess::ComputeLabelMatrixInternal: no ouput labels defined.");
    }
    unsigned d = labels[0].size();
    Y.resize(n,d);

#pragma omp parallel for
    for(unsigned i=0; i<n; i++){
        Y.block(i,0,1,d) = labels[i].adjoint();
    }
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeRegressionVectors(){

    // Computation of kernel matrix
    if(debug){
        std::cout << "GaussianProcess::ComputeRegressionVectors: calculating regression vectors... " << std::endl;
    }

    // compute the core matrix which is inv(K + sigma I)
    // This is a separate function call because it is also used if the core matrix
    // is not stored due to the efficient storage setting
    ComputeCoreMatrix(m_CoreMatrix);

    // calculate label matrix
    // TODO: if a mean support is implemented, the mean has to be subtracted from the labels!
    MatrixType Y;
    ComputeLabelMatrix(Y);


    // calculate regression vectors
    m_RegressionVectors = m_CoreMatrix * Y ; // inv(K + sigma)*Y

    // deleting core matrix if the storage has to be handled efficiently
    // - it is not used for regression
    // - but it is needed to compute the credible interval
    if(m_EfficientStorage){
        m_CoreMatrix.setZero(0,0);
    }
    if(debug){
        std::cout << "GaussianProcess::ComputeRegressionVectors: calculating regression vectors [done]" << std::endl;
    }
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeKernelVector(const typename GaussianProcess<TScalarType>::VectorType &x,
                                                       typename GaussianProcess<TScalarType>::VectorType &Kx) const{
    if(!m_Initialized){
        throw std::string("GaussianProcess::ComputeKernelVectorInternal: gaussian process is not initialized.");
    }
    ComputeKernelVectorInternal(x, Kx, m_SampleVectors);
}

template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeKernelVectorInternal(const typename GaussianProcess<TScalarType>::VectorType &x,
                                                               typename GaussianProcess<TScalarType>::VectorType &Kx,
                                                               const typename GaussianProcess<TScalarType>::VectorListType& samples) const{
    Kx.resize(samples.size());

#pragma omp parallel for
    for(unsigned i=0; i<Kx.size(); i++){
        Kx(i) = (*m_Kernel)(x, samples[i]);
    }
}


template< class TScalarType >
void GaussianProcess<TScalarType>::ComputeDifferenceMatrix(const typename GaussianProcess<TScalarType>::VectorType &x,
                                                           typename GaussianProcess<TScalarType>::MatrixType &X) const{
    unsigned n = m_SampleVectors.size();
    unsigned d = x.size();
    X.resize(n,d);

    for(unsigned i=0; i<n; i++){
        X.block(i,0,1,d) = (x - m_SampleVectors[i]).adjoint();
    }
}

template< class TScalarType >
void GaussianProcess<TScalarType>::CheckInputDimension(const typename GaussianProcess<TScalarType>::VectorType &x, std::string msg_prefix) const{
    if(x.size()!=m_InputDimension){
        std::stringstream error_msg;
        error_msg << msg_prefix << "dimension of input vector ("<< x.size() << ") does not correspond to the input dimension (" << m_InputDimension << ").";
        throw std::string(error_msg.str());
    }
}

template< class TScalarType >
void GaussianProcess<TScalarType>::CheckOutputDimension(const typename GaussianProcess<TScalarType>::VectorType &y, std::string msg_prefix) const{
    if(y.size()!=m_OutputDimension){
        std::stringstream error_msg;
        error_msg << msg_prefix << "dimension of output vector ("<< y.size() << ") does not correspond to the output dimension (" << m_OutputDimension << ").";
        throw std::string(error_msg.str());
    }
}


}

template class gpr::GaussianProcess<float>;
template class gpr::GaussianProcess<double>;
