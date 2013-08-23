#ifndef ADSAI_EXTERNAL_PARAM
#define ADSAI_EXTERNAL_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/trws/trws_adsal.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAdsai{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAdsai<INFERENCE> SelfType;


   static void set 
   (
      Parameter & p,
      const size_t numberOfIterations
   ) {
      p.maxNumberOfIterations()=numberOfIterations;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str(),init<>())
         .def ("set", &SelfType::set, 
            (
               boost::python::arg("steps")=1000
            )
         )
         .def_readwrite("steps",&Parameter::maxNumberOfIterations, "number of iterations")
      ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::ADSal<GM,ACC> >  
: public  InfParamExporterAdsai<opengm::ADSal<GM,ACC> > {

};

#endif