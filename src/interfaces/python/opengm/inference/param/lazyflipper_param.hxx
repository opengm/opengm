#ifndef LAZY_FLIPPER_PARAM
#define LAZY_FLIPPER_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/lazyflipper.hxx>

using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterLazyFlipper{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterLazyFlipper<DEPTH,INFERENCE> SelfType;

   inline static void set 
   (
      Parameter & p,
      const size_t maxSubgraphSize
   ) {
      p.maxSubgraphSize_=maxSubgraphSize;
   } 

   void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
      class_<Parameter > ( className.c_str() , init< const size_t > (args("maxSubGraphSize")))
      .def(init<>())
      .def_readwrite("maxSubgraphSize", &Parameter::maxSubgraphSize_,
      "maximum subgraph size which is optimized"
      )
      .def ("set", &SelfType::set, 
         (
            arg("maxSubgraphSize")=2
         ) 
      ,
      "Set the parameters values.\n\n"
      "All values of the parameter have a default value.\n\n"
      "Args:\n\n"
      "  maxSubgraphSize: maximum subgraph size which is optimized\n\n"
      "Returns:\n"
      "  None\n\n"
      )
   ;
   }
};

template<class DEPTH,class GM,class ACC>
class InfParamExporter<DEPTH,opengm::LazyFlipper<GM,ACC> >  : public  InfParamExporterLazyFlipper<DEPTH,opengm::LazyFlipper< GM,ACC> > {

};

#endif