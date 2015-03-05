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

#include <map>
#include <string>
#include <memory>

/*
 *  Factory Methods and Class
 *
 *  - as soon a kernel class is registered in the kernel factory
 *      the load function must be implemented.
 */

namespace gpr{


// Factory methods which are stored as function pointers
// in the map of KernelFactory. These methods are called
// if one loads a kernel from the KernelFactory.
template<typename TKernel>
std::shared_ptr< Kernel<float> > Create(const Kernel<float>::ParameterVectorType& parameters){
    return TKernel::Load(parameters);
}

template<typename TKernel>
std::shared_ptr< Kernel<double> > Create(const Kernel<double>::ParameterVectorType& parameters){
    return TKernel::Load(parameters);
}

// Factory class
template <class TScalarType>
class KernelFactory{
public:
    typedef Kernel<TScalarType> KernelType;
    typedef std::shared_ptr<KernelType> KernelTypePointer;
    typedef KernelTypePointer (*ComponentFactoryFuncPtr)(const typename KernelType::ParameterVectorType&); // function pointer to Create<TKernel>
    typedef std::map<const std::string, ComponentFactoryFuncPtr> MapType;

    static std::shared_ptr<MapType> m_Map;

public:


    template< typename TKernel >
    static void AddType(const std::string& componentName){
        ComponentFactoryFuncPtr function = &Create<TKernel>;
        m_Map->insert(std::make_pair(componentName, function));
    }

    static KernelTypePointer Load(const std::string& kernel_string, const typename KernelType::ParameterVectorType& parameters){
        typename MapType::iterator it = m_Map->find(kernel_string);
        return (*it->second)(parameters);
    }

    static void RegisterKernels(){
        KernelFactory<TScalarType>::AddType< WhiteKernel<TScalarType> >(std::string("WhiteKernel"));
        KernelFactory<TScalarType>::AddType< GaussianKernel<TScalarType> >(std::string("GaussianKernel"));
        KernelFactory<TScalarType>::AddType< PeriodicKernel<TScalarType> >(std::string("PeriodicKernel"));
        KernelFactory<TScalarType>::AddType< RationalQuadraticKernel<TScalarType> >(std::string("RationalQuadraticKernel"));
        KernelFactory<TScalarType>::AddType< SumKernel<TScalarType> >(std::string("SumKernel"));
        KernelFactory<TScalarType>::AddType< ProductKernel<TScalarType> >(std::string("ProductKernel"));
    }

    // Loads/Builds a kernel out of a kernel string
    static KernelTypePointer GetKernel(std::string& kernel_string){

        // get kernel name
        std::stringstream line_stream(kernel_string);
        std::string kernel_type;

        if(!std::getline(line_stream, kernel_type, '(')){
            throw std::string("KernelFactory::GetKernel: failed to tokanize kernel name string");
        }

        if(kernel_type.compare("SumKernel")==0){
            kernel_string = kernel_string.substr(std::string("SumKernel(").size());

            KernelTypePointer k1 = GetKernel(kernel_string);

            int pos = kernel_string.find("),");
            if(pos == std::string::npos) throw std::string("KernelFactory::GetKernel: failed to tokanize  sum kernel name string");
            kernel_string = kernel_string.substr(pos+2);

            KernelTypePointer k2 = GetKernel(kernel_string);

            typedef SumKernel<TScalarType>          KernelType;
            typedef std::shared_ptr<KernelType>     KernelTypePointer;
            KernelTypePointer k(new KernelType(k1,k2));
            return k;
        }

        if(kernel_type.compare("ProductKernel")==0){
            kernel_string = kernel_string.substr(std::string("ProductKernel(").size());

            KernelTypePointer k1 = GetKernel(kernel_string);

            int pos = kernel_string.find("),");
            if(pos == std::string::npos) throw std::string("KernelFactory::GetKernel: failed to tokanize  product kernel name string");
            kernel_string = kernel_string.substr(pos+2);

            KernelTypePointer k2 = GetKernel(kernel_string);

            typedef ProductKernel<TScalarType>          KernelType;
            typedef std::shared_ptr<KernelType>     KernelTypePointer;
            KernelTypePointer k(new KernelType(k1,k2));
            return k;
        }


        // get kernel parameters
        typename KernelType::ParameterVectorType kernel_parameters;
        do{
            typename KernelType::ParameterType p;
            if(std::getline(line_stream, p, ',')){
                if(p.find(")") != std::string::npos) break;
                kernel_parameters.push_back(p);
            }
            else{
                break;
            }
        }while(true);


        if(kernel_type.compare("GaussianKernel")==0){
            typedef GaussianKernel<TScalarType>		KernelType;
            typedef std::shared_ptr<KernelType>     KernelTypePointer;

            KernelTypePointer k = std::dynamic_pointer_cast<KernelType>(KernelFactory<TScalarType>::Load(kernel_type, kernel_parameters));
            return k;
        }
        else if(kernel_type.compare("PeriodicKernel")==0){
            typedef PeriodicKernel<TScalarType>		KernelType;
            typedef std::shared_ptr<KernelType>     KernelTypePointer;

            KernelTypePointer k = std::dynamic_pointer_cast<KernelType>(KernelFactory<TScalarType>::Load(kernel_type, kernel_parameters));
            return k;
        }
        else if(kernel_type.compare("RationalQuadraticKernel")==0){
            typedef RationalQuadraticKernel<TScalarType>		KernelType;
            typedef std::shared_ptr<KernelType>                 KernelTypePointer;

            KernelTypePointer k = std::dynamic_pointer_cast<KernelType>(KernelFactory<TScalarType>::Load(kernel_type, kernel_parameters));
            return k;
        }
        else if(kernel_type.compare("WhiteKernel")==0){
            typedef WhiteKernel<TScalarType>		KernelType;
            typedef std::shared_ptr<KernelType>     KernelTypePointer;

            KernelTypePointer k = std::dynamic_pointer_cast<KernelType>(KernelFactory<TScalarType>::Load(kernel_type, kernel_parameters));
            return k;
        }

        throw std::string("KernelFactory::GetKernel: failed to load kernel.");
    }
};


//template<> KernelFactory<float>::MapTyp KernelFactory<float>::m_Map();

template<> std::shared_ptr< std::map<const std::string, std::shared_ptr< Kernel<float> > (*)(const typename Kernel<float>::ParameterVectorType&)> > KernelFactory<float>::m_Map( new std::map<const std::string, std::shared_ptr< Kernel<float> > (*)(const typename Kernel<float>::ParameterVectorType&)>());
template<> std::shared_ptr< std::map<const std::string, std::shared_ptr< Kernel<double> > (*)(const typename Kernel<double>::ParameterVectorType&)> > KernelFactory<double>::m_Map( new std::map<const std::string, std::shared_ptr< Kernel<double> > (*)(const typename Kernel<double>::ParameterVectorType&)>());

template class KernelFactory<float>;
template class KernelFactory<double>;

}

