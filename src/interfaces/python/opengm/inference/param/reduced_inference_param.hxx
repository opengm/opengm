#ifndef REDUCED_INFERENCE_PARAM
#define REDUCED_INFERENCE_PARAM

#include <opengm/inference/reducedinference.hxx>
#include "param_exporter_base.hxx"
//solver specific


using namespace boost::python;

template<class INFERENCE>
class InfParamExporterReducedInference{

public:
    typedef typename INFERENCE::ValueType   ValueType;
    typedef typename INFERENCE::Parameter   Parameter;
    typedef typename INFERENCE::InfType     SubInfType;
    typedef typename SubInfType::Parameter  SubInfParam;
    typedef InfParamExporterReducedInference<INFERENCE> SelfType;

    inline static void set
    (
        Parameter & p,
        const SubInfParam & subInfParam,
        bool persistency,
        bool tentacle,
        bool connectedComponents
    ){
        p.subParameter_=subInfParam;
        p.Persistency_=persistency;
        p.Tentacle_=tentacle;
        p.ConnectedComponents_=connectedComponents;
    }

    void static exportInfParam(const std::string & className){
        
        class_<Parameter >( className.c_str(),DefaultParamStr::classDocStr().c_str(), init<>( DefaultParamStr::emptyConstructorDocStr().c_str() )) 
        .def ("set", &SelfType::set,
            (
                boost::python::arg("subInfParam")=SubInfParam(),
                boost::python::arg("persistency")=false,
                boost::python::arg("tentacle")=false,
                boost::python::arg("connectedComponents")=false
            )
        ) 
        .def_readwrite( "subInfParam",&Parameter::subParameter_,"use reduction by removing tentacles")
        .def_readwrite( "persistency",&Parameter::Persistency_,"use reduction persistency")
        .def_readwrite( "tentacle", &Parameter::Tentacle_,"use reduction by removing tentacles")
        .def_readwrite( "connectedComponents",&Parameter::ConnectedComponents_,"use reduction by finding connect components")
        ; 
    }
};

template<class GM,class ACC,class SUB_INF>
class InfParamExporter<opengm::ReducedInference<GM,ACC,SUB_INF> > 
 : public InfParamExporterReducedInference< opengm::ReducedInference< GM,ACC,SUB_INF>  > {
};

#endif