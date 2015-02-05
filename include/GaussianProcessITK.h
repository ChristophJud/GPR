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

#ifndef GaussianProcessITK_h
#define GaussianProcessITK_h

#include <string>
#include <vector>
#include <memory>

#include <vnl/vnl_vector.h>
#include <vnl/vnl_matrix.h>

#include "GaussianProcess.h"

template< class TScalarType >
class GaussianProcessITK
{
public:
	typedef GaussianProcessITK Self;
	typedef GaussianProcess<TScalarType> GaussianProcessType;
	typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

	typedef typename GaussianProcessType::VectorType	VectorType;
	typedef typename GaussianProcessType::MatrixType	MatrixType;

	typedef vnl_vector<TScalarType> ITKVectorType;
	typedef vnl_matrix<TScalarType> ITKMatrixType;

	/*
	 * Add a new sample lable pair to the gaussian process
	 *  x is the input vector
	 *  y the corresponding label vector
	 */
	void AddSample(const ITKVectorType &x, const ITKVectorType &y){
		m_GaussianProcess->AddSample(this->ConvertVector(x),this->ConvertVector(y));
	}

	/*
	 * Predict new data point
	 */
	ITKVectorType Predict(const ITKVectorType &x){
		return this->ConvertVectorBack(m_GaussianProcess->Predict(this->ConvertVector(x)));
	}

	/*
	 * Predict new point (return value) and its derivative input parameter D
	 */
	ITKVectorType PredictDerivative(const ITKVectorType &x, ITKMatrixType &D){
		MatrixType M;
		ITKVectorType v = this->ConvertVectorBack(m_GaussianProcess->PredictDerivative(this->ConvertVector(x),M));
		D.set_size(M.rows(),M.cols());
		for(unsigned i=0; i<M.rows(); i++){
			for(unsigned j=0; j<M.cols(); j++){
				D(i,j) = M(i,j);
			}
		}
		return v;
	}

	TScalarType GetSigma(){
		return m_GaussianProcess->GetSigma();
	}

	void SetSigma(TScalarType sigma){
		m_GaussianProcess->SetSigma(sigma);
	}

	unsigned GetNumberOfInputDimensions(){
		return m_GaussianProcess->GetNumberOfInputDimensions();
	}

	inline TScalarType operator()(const ITKVectorType & x, const ITKVectorType & y) const{
		return (*m_GaussianProcess)(this->ConvertVector(x), this->ConvertVector(y));
	}

	/*
	 * If sample data has changed perform learning step
	 */
	void Initialize(){
		m_GaussianProcess->Initialize();
	}

	GaussianProcessITK(GaussianProcessTypePointer gp){
		m_GaussianProcess = gp;
	}
	virtual ~GaussianProcessITK() {}

	void DebugOn(){
		m_GaussianProcess->DebugOn();
	}

	unsigned GetNumberOfSamples(){
		return m_GaussianProcess->GetNumberOfSamples();
	}

	void Save(std::string prefix){
		m_GaussianProcess->Save(prefix);
	}

	void Load(std::string prefix){
		m_GaussianProcess->Load(prefix);
	}

	void ToString(){
		m_GaussianProcess->ToString();
	}

private:
	GaussianProcessTypePointer m_GaussianProcess;

	VectorType ConvertVector(ITKVectorType x){
		VectorType y = VectorType::Zero(x.size());
		for(unsigned i=0; i<x.size(); i++){
			y[i] = x[i];
		}
		return y;
	}
	ITKVectorType ConvertVectorBack(VectorType x){
		ITKVectorType y(x.size());
		for(unsigned i=0; i<x.size(); i++){
			y[i] = x[i];
		}
		return y;
	}

	MatrixType ConvertMatrix(ITKMatrixType m){
		MatrixType y = MatrixType::Zero(m.rows(),m.cols());
		for(unsigned i=0; i<m.rows(); i++){
			for(unsigned j=0; j<m.cols(); j++){
				y(i,j) = m(i,j);
			}
		}
		return y;
	}

	ITKMatrixType ConvertMatrixBack(MatrixType m){
		ITKMatrixType y(m.rows(),m.cols());
		for(unsigned i=0; i<m.rows(); i++){
			for(unsigned j=0; j<m.cols(); j++){
				y(i,j) = m(i,j);
			}
		}
		return y;
	}

	GaussianProcessITK(const Self &); //purposely not implemented
	void operator=(const Self &); //purposely not implemented
};

#endif
