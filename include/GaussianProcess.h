#ifndef GaussianProcess_h
#define GaussianProcess_h

#include <fstream>
#include <string>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <Eigen/Cholesky>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/filesystem.hpp>
namespace fs = boost::filesystem;

#include "Kernel.h"
#include "MatrixIO.h"


template< class TScalarType >
class GaussianProcess
{
public:
	typedef GaussianProcess Self;
	typedef Kernel<TScalarType> KernelType;
	typedef std::shared_ptr<KernelType> KernelTypePointer;
	
    typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, 1> VectorType;
    typedef Eigen::Matrix<TScalarType, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
    typedef Eigen::DiagonalMatrix<TScalarType, Eigen::Dynamic> DiagMatrixType;

	typedef std::shared_ptr<MatrixType> MatrixTypePointer;

	typedef std::vector<VectorType> VectorListType;
	typedef std::vector<MatrixType> MatrixListType;
	
	/*
	 * Add a new sample lable pair to the gaussian process
	 *  x is the input vector
	 *  y the corresponding label vector
	 */
	void AddSample(const VectorType &x, const VectorType &y){
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

	/*
	 * Predict new data point
	 */
	VectorType Predict(const VectorType &x){
		Initialize();
		CheckInputDimension(x, "GaussianProcess::Predict: ");
		VectorType Kx;
		ComputeKernelVector(x, Kx);
		return (Kx.adjoint() * m_RegressionVectors).adjoint();
	}

	/*
	 * Predict new point (return value) and its derivative input parameter D
	 */
	VectorType PredictDerivative(const VectorType &x, MatrixType &D){
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

	TScalarType GetSigma(){
		return m_Sigma;
	}

	void SetSigma(TScalarType sigma){
		m_Sigma = sigma;
		m_Initialized = false;
	}

	unsigned GetNumberOfInputDimensions(){ return m_InputDimension; }

	inline TScalarType operator()(const VectorType & x, const VectorType & y) const{
		throw std::string("GaussianProcess::Initialize: not implemented yet");
	}
	
	/*
	 * If sample data has changed perform learning step
	 */
	void Initialize(){
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

	GaussianProcess(KernelTypePointer kernel) : m_Sigma(0), m_Initialized(false), m_InputDimension(0), m_OutputDimension(0), debug(false) {
		m_Kernel = kernel;
	}
	virtual ~GaussianProcess() {}

	void DebugOn(){
		debug = true;
	}

	unsigned GetNumberOfSamples(){
		return m_SampleVectors.size();
	}

	void Save(std::string prefix){
		if(!m_Initialized){
			throw std::string("GaussianProcess::Save: gaussian process is not initialized.");
		}

		if(debug){
			std::cout << "GaussianProcess::Save: writing gaussian process: " << std::endl;
			std::cout << "\t " << prefix+"-RegressionVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-SampleVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-LabelVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-ParameterFile.txt" << std::endl;
		}

		// save regression vectors
		WriteMatrix<MatrixType>(m_RegressionVectors, prefix+"-RegressionVectors.txt");

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
		// KernelType, KernelParameter, noise, InputDimension, OutputDimension
		std::ofstream parameter_outfile;
		parameter_outfile.open(std::string(prefix+"-ParameterFile.txt").c_str());
		parameter_outfile << m_Kernel->ToString() << " " << m_Kernel->GetParameter() << " " << m_Sigma << " " << m_InputDimension << " " << m_OutputDimension;
		parameter_outfile.close();
	}

	void Load(std::string prefix){
		if(debug){
			std::cout << "GaussianProcess::Load: loading gaussian process: " << std::endl;
			std::cout << "\t " << prefix+"-RegressionVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-SampleVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-LabelVectors.txt" << std::endl;
			std::cout << "\t " << prefix+"-ParameterFile.txt" << std::endl;
		}

		// load sample vectors
		std::string rv_filename = prefix+"-RegressionVectors.txt";
		fs::path rv_file(rv_filename.c_str());
		if(!(fs::exists(rv_file) && !fs::is_directory(rv_file))){
			throw std::string("GaussianProcess::Load: "+rv_filename+" does not exist or is a directory.");
		}
		m_RegressionVectors = ReadMatrix<MatrixType>(rv_filename);

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

		std::string kernel_type;
		TScalarType kernel_parameter;

		// reading parameter file
		std::string line;
		if(std::getline(parameter_infile, line)) {
			std::stringstream line_stream(line);
			if(!(line_stream >> kernel_type &&
					line_stream >> kernel_parameter &&
					line_stream >> m_Sigma &&
					line_stream >> m_InputDimension &&
					line_stream >> m_OutputDimension)){
				throw std::string("GaussianProcess::Load: parameter file is corrupt");
			}
		}
		parameter_infile.close();

		typedef GaussianKernel<TScalarType>		KernelType;
		typedef std::shared_ptr<KernelType> KernelTypePointer;
		KernelTypePointer k(new KernelType(kernel_parameter));
		m_Kernel = k;

		m_Initialized = true;
	}

	void ToString(){
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
		std::cout << "       - Parameter:\t" << m_Kernel->GetParameter() << std::endl;
		std::cout << "---------------------------------------" << std::endl;
	}

private:
	/*
	 * Computation of kernel matrix K_ij = k(x_i, x_j)
	 * 	- it is symmetric therefore only half of the kernel evaluations
	 * 	  has to be performed
	 */
	void ComputeKernelMatrix(MatrixType &M){
		unsigned n = m_SampleVectors.size();
		M.resize(n,n);

		#pragma omp parallel for
		for(unsigned i=0; i<n; i++){
			for(unsigned j=i; j<n; j++){
				TScalarType v = (*m_Kernel)(m_SampleVectors[i],m_SampleVectors[j]);
				M(i,j) = v;
				M(j,i) = v;
			}
		}
	}

	/*
	 * Bring the label vectors in a matrix form Y,
	 * where the rows are the labels.
	 */
	void ComputeLabelMatrix(MatrixType &Y){
		unsigned n = m_LabelVectors.size();
		if(!(n > 0)){
			throw std::string("GaussianProcess::ComputeRegressionVectors: no ouput labels defined during computation of the regression vectors.");
		}
		unsigned d = m_LabelVectors[0].size();
		Y.resize(n,d);

		#pragma omp parallel for
		for(unsigned i=0; i<n; i++){
			Y.block(i,0,1,d) = m_LabelVectors[i].adjoint();
		}
	}

	/*
	 * Lerning is performed.
	 */
	void ComputeRegressionVectors(){

		// Computation of kernel matrix
		if(debug){
			std::cout << "GaussianProcess::ComputeRegressionVectors: building kernel matrix... ";
			std::cout.flush();
		}
		MatrixType K;
		ComputeKernelMatrix(K);
		if(debug) std::cout << "[done]" << std::endl;


		// add noise variance to diagonal
		for(unsigned i=0; i<K.rows(); i++){
			K(i,i) += m_Sigma;
		}

		// calculate label matrix
		MatrixType Y;
		ComputeLabelMatrix(Y);


		// calculate regression vectors
		if(debug){
			std::cout << "GaussianProcess::ComputeRegressionVectors: inverting kernel matrix... ";
			std::cout.flush();
		}
		m_RegressionVectors = K.inverse() * Y ;
		if(debug) std::cout << "[done]" << std::endl;
	}

	/*
	 * Computation of the kernel vector V_i = k(x, x_i)
	 */
	void ComputeKernelVector(const VectorType &x, VectorType &Kx){
		if(!m_Initialized){
			throw std::string("GaussianProcess::ComputeKernelVecotr: gaussian process is not initialized.");
		}
		Kx.resize(m_RegressionVectors.rows());

		#pragma omp parallel for
		for(unsigned i=0; i<Kx.size(); i++){
			Kx(i) = (*m_Kernel)(x, m_SampleVectors[i]);
		}
	}


	/*
	 * Compute difference matrix X = [x-x_0, x-x_1, ... x-x_n]^T
	 */
	void ComputeDifferenceMatrix(const VectorType &x, MatrixType &X){
		unsigned n = m_SampleVectors.size();
		unsigned d = x.size();
		X.resize(n,d);

		for(unsigned i=0; i<n; i++){
			X.block(i,0,1,d) = (x - m_SampleVectors[i]).adjoint();
		}
	}

	/*
	 * Assertion functions to check input and output dimensions of the vectors
	 */
	void CheckInputDimension(const VectorType &x, std::string msg_prefix){
		if(x.size()!=m_InputDimension){
			std::stringstream error_msg;
			error_msg << msg_prefix << "dimension of input vector ("<< x.size() << ") does not correspond to the input dimension (" << m_InputDimension << ").";
			throw std::string(error_msg.str());
		}
	}
	void CheckOutputDimension(const VectorType &y, std::string msg_prefix){
		if(y.size()!=m_OutputDimension){
			std::stringstream error_msg;
			error_msg << msg_prefix << "dimension of output vector ("<< y.size() << ") does not correspond to the output dimension (" << m_OutputDimension << ".";
			throw std::string(error_msg.str());
		}
	}

	KernelTypePointer m_Kernel;
	
	TScalarType m_Sigma;

	VectorListType m_SampleVectors;  // Dimensionality: TInputDimension
	VectorListType m_LabelVectors;   // Dimensionality: TOutputDimension
	MatrixType m_RegressionVectors; // for each output dimension there is one regression vector
	
	bool m_Initialized;
	unsigned m_InputDimension;
	unsigned m_OutputDimension;

	bool debug;

	GaussianProcess(const Self &); //purposely not implemented
	void operator=(const Self &); //purposely not implemented
};



#endif
