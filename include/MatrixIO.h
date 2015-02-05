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
