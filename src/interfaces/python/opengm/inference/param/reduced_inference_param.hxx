#ifndef REDUCED_INFERENCE_PARAM
#define REDUCED_INFERENCE_PARAM

#include <opengm/inference/reducedinference.hxx>
#include "param_exporter_base.hxx"
//solver specific


using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterReducedInference{

public:
    typedef typename INFERENCE::ValueType ValueType;
    typedef typename INFERENCE::Parameter Parameter;
    typedef InfParamExporterReducedInference<DEPTH,INFERENCE> SelfType;

    inline static void set
    (
        Parameter & p//,
        //double scale     
    ){
        //p.scale_=scale;
    }

    void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
        
        class_<Parameter >( className.c_str(),DefaultParamStr::classDocStr().c_str(), init<>( DefaultParamStr::emptyConstructorDocStr().c_str() )) 
          .def ("set", &SelfType::set) 
          //.def_readwrite( "scale", & Parameter::scale_)
          ; 
    }
};

template<class DEPTH,class GM,class ACC,class SUB_INF>
class InfParamExporter<DEPTH,opengm::ReducedInference<GM,ACC,SUB_INF> > 
 : public  
  InfParamExporterReducedInference
  <
    DEPTH,
    opengm::ReducedInference< GM,ACC,SUB_INF> 
  > 

 {

};

#endif