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

#include <unsupported/Eigen/FFT>

#include "GaussianProcess.h"

namespace gpr{

// Period length estimation based on fast fourier transfrom
// - with ommit, the first few amplitues can be ignored
// - the function can be used to roughly estimate the period
//   hyperparameter of the periodic kernel
template<class TScalarType>
TScalarType GetLocalPeriodLength(const typename GaussianProcess<TScalarType>::VectorType &vec, unsigned ommit=1){
    unsigned interval_size = vec.rows();

    Eigen::FFT<float> fft;

    std::vector<float> t;
    for(unsigned i=0; i<interval_size; i++){
        t.push_back(vec[i]);
    }

    std::vector<std::complex<float> > f;
    fft.fwd(f, t);

    unsigned max_index = 0;
    float max = std::numeric_limits<float>::lowest();
    for(unsigned i=ommit; i<interval_size/2; i++){
        float amp = std::abs(f[i]);
        if(amp>max){
            max = amp;
            max_index = i;
        }
    }

    return static_cast<TScalarType>(interval_size) / static_cast<TScalarType>(max_index);
}

}
