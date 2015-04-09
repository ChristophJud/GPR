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

#include <vector>
#include <cmath>
#include <limits>

#include "Prior.h"

namespace gpr{

// Sampling density p using the inverse transform method:
// - u ~ U(0,1)
// - F(x) = icdf(F)(u)
// However, we use equidistant samples in the interval [0,1]
// Attention, there might be more or less samples in the return vector than num_points
template<class TScalarType>
std::vector<TScalarType> GetSamples(const Density<TScalarType>* p, unsigned num_points){

    TScalarType x_start = std::max(std::numeric_limits<TScalarType>::epsilon(), p->mode()-std::sqrt(p->variance()));
    TScalarType x_end = p->mode()+std::sqrt(p->variance());

    std::vector<TScalarType> distributed_vector;
    distributed_vector.push_back(p->mode());
    if(num_points==0) return distributed_vector;

    // generating uniform sampled vector
    //std::uniform_real_distribution<ScalarType> U(0,1); // for uniform samples

    for(unsigned num_samples=0; num_samples<=num_points; num_samples++){
        TScalarType u = 1.0/num_points * num_samples; // equidistant sample
        TScalarType d = p->icdf(u); // sample of distribution

        // only accept samples within interval of mode+-sqrt(variance)
        if(d >= x_start && d <= x_end){
            distributed_vector.push_back(d); // accept sample
            //num_samples++;
        }
    }
    return distributed_vector;
}

}
