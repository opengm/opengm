#ifndef INFERENCE_PARAM_EXPORTER
#define INFERENCE_PARAM_EXPORTER

#include <string>
#include <boost/python.hpp>

template<int DEPTH>
class SubExportDepth;

namespace exportTag{
    typedef SubExportDepth<0>   NonRecursive;
    typedef SubExportDepth<-1>  Recursive;
    typedef NonRecursive        NoSubInf;
}

template<class DEPTH,class INFERENCE>
class InfParamExporter{

    void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){

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


//http://stackoverflow.com/questions/12929196/how-to-speed-up-boostpythonextract-when-passing-a-list-from-python-to-a-c


template<class DEPTH,class INF>
void exportInfParam(
    const std::string & className,
    const std::vector<std::string> & subInfParamNames=std::vector<std::string>()
){
    //append_subnamespace("solver");
    {
        append_subnamespace("parameter");
        InfParamExporter<DEPTH,INF>::exportInfParam(className,subInfParamNames);
    }
    
}


#endif //INFERENCE_PARAM_EXPORTER