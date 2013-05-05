#ifndef INFERENCE_PARAM_EXPORTER
#define INFERENCE_PARAM_EXPORTER

#include <string>
#include <boost/python.hpp>

template<int DEPTH>
class SubExportDepth;



template<class INFERENCE>
class InfParamExporter{

    void static exportInfParam(const std::string & className){

    }
};


template<class CLASS,class T>
inline T  selfReturner(const CLASS & c,const T & t){
    return t;
}

class DefaultParamStr{
public:
    static std::string classDocStr(){
        return "Parameter Object for an opengm Inference Object";
    }
    static std::string emptyConstructorDocStr(){
        return "Parameter Object for an opengm Inference Object";
    }
};

using namespace boost::python;


#define append_subnamespace(SCOPE_NAME) \
    const std::string pySubNamespaceName(SCOPE_NAME); \
    scope current; \
    std::string currentScopeName(extract<const char*>(current.attr("__name__"))); \
    std::string submoduleName = currentScopeName + std::string(".") + pySubNamespaceName; \
    object submodule(borrowed(PyImport_AddModule(submoduleName.c_str()))); \
    current.attr(pySubNamespaceName.c_str()) = submodule; \
    submodule.attr("__package__") = submoduleName.c_str(); \
    scope submoduleScope = submodule


template<class INF>
void exportInfParam(
    const std::string & className
){
    {
        append_subnamespace("parameter");
        InfParamExporter<INF>::exportInfParam(className);
    }
    
}


#endif //INFERENCE_PARAM_EXPORTER