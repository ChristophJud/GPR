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
#include <utility> // std::tuple

#include <unsupported/Eigen/FFT>

#include "GaussianProcess.h"

namespace gpr{

// Period length estimation based on fast fourier transfrom
// - with ommit, the first few amplitues can be ignored
// - the function can be used to roughly estimate the period
//   hyperparameter of the periodic kernel
template<class TScalarType>
std::tuple<TScalarType, TScalarType, TScalarType> // (period length, dominant amplitude, sinuns likeness)
GetLocalPeriodLength(const typename GaussianProcess<TScalarType>::VectorType &vec, unsigned ommit=1){
    if(vec.rows() < 4+ommit) throw std::string("GetLocalPeriodLength: longer signal required. Check if a column vector is provided!");
    unsigned interval_size = vec.rows();

    Eigen::FFT<float> fft;

    std::vector<float> t;
    for(unsigned i=0; i<interval_size; i++){
        t.push_back(vec[i]);
    }

    std::vector<std::complex<float> > f;
    fft.fwd(f, t);

    unsigned max_index = 0;
    float amp_max = std::numeric_limits<float>::lowest();
    float amp_integral = 0;
    for(unsigned i=ommit; i<interval_size/2; i++){
        float amp = 2*std::abs(f[i])/interval_size;
        if(amp>amp_max){
            amp_max = amp;
            max_index = i;
        }
        amp_integral += amp;
    }

    //std::cout << "amp max: " << amp_max << ", amp integral: " << amp_integral << std::endl;

    // returns a tuple of
    // - number of indices per period
    // - the maximum amplitude
    // - ration between integral over amps and max amp
    double period_length = static_cast<TScalarType>(interval_size) / static_cast<TScalarType>(max_index);
    double amplitude = amp_max;
    double sinus_likeness;
    if(amp_integral-amp_max < std::numeric_limits<TScalarType>::min()){
        sinus_likeness = std::numeric_limits<TScalarType>::max();
    }
    else{
        sinus_likeness = (amp_integral/(amp_integral-amp_max) - 1); // multiplied by 2 becaus only half of the spectrum is integrated
    }
    return std::make_tuple(period_length, amplitude, sinus_likeness);
}

}
