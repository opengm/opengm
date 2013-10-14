#ifndef LAZY_FLIPPER_PARAM
#define LAZY_FLIPPER_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/pbp.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterPbp{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterPbp<INFERENCE> SelfType;

   inline static void set 
   (
      Parameter & p,
      const size_t steps,
      const ValueType pruneLimit
   ) {
      p.steps_=steps;
      p.pruneLimit_=pruneLimit;
   } 

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str() , init< > ())
      .def_readwrite("steps", &Parameter::steps_,
      "maximum iterations / steps"
      )
      .def_readwrite("pruneLimit", &Parameter::pruneLimit_,
      "label prune limit (limit on relative beliefe"
      )
      .def ("set", &SelfType::set, 
         (
            boost::python::arg("steps")=10,
            boost::python::arg("pruneLimit")=0.2
         ) 
      )
   ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::PBP<GM,ACC> >  : public  InfParamExporterPbp<opengm::PBP< GM,ACC> > {

};

#endif