#ifndef LAZY_FLIPPER_PARAM
#define LAZY_FLIPPER_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/lazyflipper.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterLazyFlipper{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLazyFlipper<INFERENCE> SelfType;

   inline static void set 
   (
      Parameter & p,
      const size_t maxSubgraphSize
   ) {
      p.maxSubgraphSize_=maxSubgraphSize;
   } 

   void static exportInfParam(const std::string & className){
      class_<Parameter > ( className.c_str() , init< > ())
      .def_readwrite("maxSubgraphSize", &Parameter::maxSubgraphSize_,
      "maximum subgraph size which is optimized"
      )
      .def ("set", &SelfType::set, 
         (
            boost::python::arg("maxSubgraphSize")=2
         ) 
      )
   ;
   }
};

template<class GM,class ACC>
class InfParamExporter<opengm::LazyFlipper<GM,ACC> >  : public  InfParamExporterLazyFlipper<opengm::LazyFlipper< GM,ACC> > {

};

#endif