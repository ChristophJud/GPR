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

#include <memory>
#include "Kernel.h"

namespace gpr {

/*
 *  TODO: describe kernel
 *
 *  p0 = scale of k1
 *  p1 = sigma of k1
 *  p2 = gauss scale of k2
 *  p3 = gauss sigma of k2
 *  p4 = period scale of k2
 *  p5 = period period of k2
 *  p6 = period sigma of k2
 *  p7 = rational scale of k3
 *  p8 = rational sigma of k3
 *  p9 = rational alpha of k3
 *  p10 = gauss scale of k4
 *  p11 = gauss sigma of k4
 *  p12 = white scale of k4
 *
 */
template<class TScalarType>
typename std::shared_ptr< Kernel<TScalarType> >
GetGeneralKernel(const std::vector<TScalarType>& params){
    typedef Kernel<TScalarType>                      KernelType;
    typedef std::shared_ptr<KernelType>         KernelTypePointer;
    typedef GaussianKernel<TScalarType>              GaussianKernelType;
    typedef std::shared_ptr<GaussianKernelType> GaussianKernelTypePointer;
    typedef WhiteKernel<TScalarType>                 WhiteKernelType;
    typedef std::shared_ptr<WhiteKernelType>    WhiteKernelTypePointer;
    typedef PeriodicKernel<TScalarType>              PeriodicKernelType;
    typedef std::shared_ptr<PeriodicKernelType> PeriodicKernelTypePointer;
    typedef RationalQuadraticKernel<TScalarType>     RationalQuadraticKernelType;
    typedef std::shared_ptr<RationalQuadraticKernelType> RationalQuadraticKernelTypePointer;
    typedef SumKernel<TScalarType>                   SumKernelType;
    typedef std::shared_ptr<SumKernelType>      SumKernelTypePointer;
    typedef ProductKernel<TScalarType>               ProductKernelType;
    typedef std::shared_ptr<ProductKernelType>  ProductKernelTypePointer;

    typedef GaussianProcess<TScalarType> GaussianProcessType;
    typedef std::shared_ptr<GaussianProcessType> GaussianProcessTypePointer;

    typedef typename GaussianProcessType::VectorType VectorType;
    typedef typename GaussianProcessType::MatrixType MatrixType;


    if(params.size() != 13){
        throw std::string("Wrong number of arguments.");
    }

    GaussianKernelTypePointer k1(new GaussianKernelType(params[1],params[0]));

    GaussianKernelTypePointer k2_gk(new GaussianKernelType(params[3],params[2]));
    PeriodicKernelTypePointer k2_pk(new PeriodicKernelType(params[4],params[5],params[6]));
    ProductKernelTypePointer k2(new ProductKernelType(k2_gk, k2_pk));

    RationalQuadraticKernelTypePointer k3(new RationalQuadraticKernelType(params[7],params[8],params[9]));

    GaussianKernelTypePointer k4_gk(new GaussianKernelType(params[11],params[10]));
    WhiteKernelTypePointer k4_wk(new WhiteKernelType(params[12]));
    SumKernelTypePointer k4(new SumKernelType(k4_gk, k4_wk));

    SumKernelTypePointer k12(new SumKernelType(k1, k2));
    SumKernelTypePointer k123(new SumKernelType(k12,k3));
    SumKernelTypePointer k1234(new SumKernelType(k123,k4));

    return k1234;
}

}
