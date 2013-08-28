#ifndef FAST_PD_PARAM
#define FAST_PD_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/external/fastPD.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterFastPd{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterFastPd<INFERENCE> SelfType;

   inline static void set 
   (
      Parameter & p,
      const size_t steps
   ) {
      p.numberOfIterations_=steps;
   } 

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str() , init< > ())
      .def_readwrite("steps", &Parameter::numberOfIterations_,
      "maximum number of iterations"
      )
      .def ("set", &SelfType::set, 
         (
            boost::python::arg("steps")=1000
         ) 
      )
   ;
   }
};

template<class GM>
class InfParamExporter<opengm::external::FastPD<GM> >  : public  InfParamExporterFastPd<opengm::external::FastPD< GM> > {

};

#endif