#ifndef MatrixIO_h
#define MatrixIO_h

#include <string>

namespace gpr{

template<typename _Matrix_Type_>
_Matrix_Type_ ReadMatrix(std::string filename);

template<typename _Matrix_Type_>
void WriteMatrix(const _Matrix_Type_& matrix, std::string filename);

bool MatrixIOTest();

}

#endif
