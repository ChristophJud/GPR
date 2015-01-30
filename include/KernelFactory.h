#ifndef KernelFactory_h
#define KernelFactory_h

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
        KernelFactory<TScalarType>::AddType< GaussianKernel<TScalarType> >(std::string("GaussianKernel"));
        KernelFactory<TScalarType>::AddType< PeriodicKernel<TScalarType> >(std::string("PeriodicKernel"));
        KernelFactory<TScalarType>::AddType< SumKernel<TScalarType> >(std::string("SumKernel"));
    }
};

//template<> KernelFactory<float>::MapTyp KernelFactory<float>::m_Map();

template<> std::shared_ptr< std::map<const std::string, std::shared_ptr< Kernel<float> > (*)(const typename Kernel<float>::ParameterVectorType&)> > KernelFactory<float>::m_Map( new std::map<const std::string, std::shared_ptr< Kernel<float> > (*)(const typename Kernel<float>::ParameterVectorType&)>());
template<> std::shared_ptr< std::map<const std::string, std::shared_ptr< Kernel<double> > (*)(const typename Kernel<double>::ParameterVectorType&)> > KernelFactory<double>::m_Map( new std::map<const std::string, std::shared_ptr< Kernel<double> > (*)(const typename Kernel<double>::ParameterVectorType&)>());

template class KernelFactory<float>;
template class KernelFactory<double>;

}

#endif
