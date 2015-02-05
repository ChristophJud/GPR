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

#include <string>
#include <iostream>
#include <fstream>
#include <memory>

#include <Eigen/Dense>

#include "MatrixIO.h"

namespace gpr{

typedef Eigen::Matrix<float, Eigen::Dynamic, 1> VectorType;
typedef Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixType;
typedef Eigen::DiagonalMatrix<float, Eigen::Dynamic> DiagMatrixType;
typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorTypeDoublePrecision;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> MatrixTypeDoublePrecision;
typedef Eigen::DiagonalMatrix<double, Eigen::Dynamic> DiagMatrixTypeDoublePrecision;

typedef unsigned long long int StreamSizeType;

template<typename _Matrix_Type_>
_Matrix_Type_ ReadMatrix(std::string filename){
	typedef typename _Matrix_Type_::Scalar MatrixElementType;
	typedef std::shared_ptr<MatrixElementType> MatrixElementTypePointer;

	std::ifstream matrix_infile;
	matrix_infile.open( filename.c_str() ); // ASCII mode

	// reading header
	std::string line;
	StreamSizeType rows, cols;
	if(std::getline(matrix_infile, line)) {
		std::stringstream line_stream(line);
		if(!(line_stream >> rows && line_stream >> cols)){
			std::stringstream error_message;
			error_message << "ReadMatrix: header is corrupt (filename " << filename << ")." << std::endl;
			throw error_message.str();
		}
	}
	matrix_infile.close();

	// reopen for binary read
	matrix_infile.open( filename.c_str(), std::ifstream::in | std::ifstream::binary );

	// read data
	unsigned header_size = line.size()+1; // +1 because of endline
	char header_buf[header_size];
	matrix_infile.read((char *)&header_buf, header_size*sizeof(char));


	// for large matrices array has to be allocated on the heap
	// a shared pointer is needed because data_buf cannot be deleted before return
	MatrixElementTypePointer data_buf(new MatrixElementType[rows*cols]);
	matrix_infile.read( (char *)&(*data_buf), rows * cols * (StreamSizeType)sizeof(MatrixElementType) );

	typename Eigen::Map<_Matrix_Type_> matrix(&(*data_buf), rows, cols);
	return matrix;
}

template<typename _Matrix_Type_>
void WriteMatrix(const _Matrix_Type_& matrix, std::string filename){
	typedef typename _Matrix_Type_::Scalar MatrixElementType;

	std::ofstream matrix_outfile;
	matrix_outfile.open( filename.c_str(), std::ios::binary );

	StreamSizeType rows = matrix.rows();
	StreamSizeType cols = matrix.cols();

	// writing header
	std::stringstream header_stream;
	header_stream << rows << " " << cols << std::endl;
	matrix_outfile.write( (char *)(header_stream.str().c_str()), header_stream.str().size() * sizeof(char) );

	// writing body
	// has to go onto the heap for large matrices
	MatrixElementType *data = new MatrixElementType[rows*cols];
	typename Eigen::Map<_Matrix_Type_>(data, rows, cols) = matrix;
	matrix_outfile.write( (char *)data, rows * cols * (StreamSizeType)sizeof(MatrixElementType) );
	matrix_outfile.close();

	delete[] data;
}

// returns true if test has been performed successfully
bool MatrixIOTest(){
	MatrixTypeDoublePrecision M = MatrixTypeDoublePrecision::Random(10,3);
	WriteMatrix<MatrixTypeDoublePrecision>(M, "/tmp/matrix.txt");

	MatrixTypeDoublePrecision N = ReadMatrix<MatrixTypeDoublePrecision>("/tmp/matrix.txt");

	if(M.size() != N.size()) return false;

	for(unsigned i=0; i<M.cols(); i++){
		for(unsigned j=0; j<M.rows(); j++){
			if(M(i,j) != N(i,j)) return false;
		}
	}
	return true;
}


template MatrixType ReadMatrix<MatrixType> (std::string);
template MatrixTypeDoublePrecision ReadMatrix<MatrixTypeDoublePrecision> (std::string);
template void WriteMatrix<MatrixType> (const MatrixType&, std::string);
template void WriteMatrix<MatrixTypeDoublePrecision> (const MatrixTypeDoublePrecision&, std::string);

}
