#ifndef DUAL_DECOMPOSITION_BUNDLE_PARAM
#define DUAL_DECOMPOSITION_BUNDLE_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterDualDecompositionBundle{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef typename INFERENCE::InfType SubInfType;
   typedef typename SubInfType::Parameter SubInfParameter;

   typedef InfParamExporterDualDecompositionBundle<INFERENCE> SelfType;



   static boost::python::tuple getSubProbParam(const Parameter & p){
      return boost::python::make_tuple(p.useAdaptiveStepsize_,p.useProjectedAdaptiveStepsize_);
   }

   static void set(
      Parameter & p
   ){


   }





   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &SelfType::set)
   ; 
   }
};

template<class GM,class SUB_INF,class BLOCK_TYPE>
class InfParamExporter<opengm::DualDecompositionBundle<GM,SUB_INF,BLOCK_TYPE> > 
 : public  InfParamExporterDualDecompositionBundle<opengm::DualDecompositionBundle< GM,SUB_INF,BLOCK_TYPE> > {

};

#endif

