#ifndef DUAL_DECOMPOSITION_MERIT_PARAM
#define DUAL_DECOMPOSITION_MERIT_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_merit.hxx>

using namespace boost::python;

template<class DEPTH,class INFERENCE>
class InfParamExporterDualDecompositionMerit{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef typename INFERENCE::InfType SubInfType;
   typedef typename SubInfType::Parameter SubInfParameter;

   typedef InfParamExporterDualDecompositionMerit<DEPTH,INFERENCE> SelfType;



   static boost::python::tuple getSubProbParam(const Parameter & p){
      return boost::python::make_tuple(p.useAdaptiveStepsize_,p.useProjectedAdaptiveStepsize_);
   }

   static void set(
      Parameter & p
   ){


   }





   void static exportInfParam(const std::string & className,const std::vector<std::string> & subInfParamNames){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def ("set", &SelfType::set)
   ; 
   }
};

template<class DEPTH,class GM,class SUB_INF,class BLOCK_TYPE>
class InfParamExporter<DEPTH,opengm::DualDecompositionMerit<GM,SUB_INF,BLOCK_TYPE> > 
 : public  InfParamExporterDualDecompositionMerit<DEPTH,opengm::DualDecompositionMerit< GM,SUB_INF,BLOCK_TYPE> > {

};

#endif

