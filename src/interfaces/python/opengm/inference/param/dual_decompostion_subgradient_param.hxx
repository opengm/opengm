#ifndef DUAL_DECOMPOSITION_PARAM
#define DUAL_DECOMPOSITION_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/graphcut.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
using namespace boost::python;

template<class INFERENCE>
class InfParamExporterDualDecompositionSubGradient{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;

   typedef typename INFERENCE::InfType SubInfType;
   typedef typename SubInfType::Parameter SubInfParameter;

   typedef InfParamExporterDualDecompositionSubGradient<INFERENCE> SelfType;

   static void setSubProbParam(Parameter & p,boost::python::tuple subProbParam){
      const size_t tupleSize=static_cast<size_t>(boost::python::len(subProbParam));
      if(tupleSize==1 || tupleSize==2){
         // 1-Entry : useAdaptiveStepsize
         boost::python::extract<bool> extractor1(subProbParam[0]);
         if(extractor1.check())
            p.useAdaptiveStepsize_= static_cast<bool >(extractor1());
         else
            throw opengm::RuntimeError("wrong data type in subProbParam tuple");
         if(tupleSize==2){
            // 2-Entry : useAdaptiveStepsize
            boost::python::extract<bool> extractor2(subProbParam[1]);
            if(extractor2.check())
               p.useProjectedAdaptiveStepsize_= static_cast<bool >(extractor2());
            else
               throw opengm::RuntimeError("wrong data type in subProbParam tuple");
         }
         else
            p.useProjectedAdaptiveStepsize_=false;
      }
      else
         throw opengm::RuntimeError(" len(subProbParam) must be at least 1 and max 2 (or subProbParam must be = None)");
   }

   static boost::python::tuple getSubProbParam(const Parameter & p){
      return boost::python::make_tuple(p.useAdaptiveStepsize_,p.useProjectedAdaptiveStepsize_);
   }

   static void set(
      Parameter & p,
      const typename Parameter::DecompositionId decompositionId,
      const size_t maximalDualOrder,
      const size_t numberOfBlocks,
      const size_t maximalNumberOfIterations,
      const double minimalAbsAccuracy,
      const double minimalRelAccuracy,
      const size_t numberOfThreads,
      const double stepsizeStride,
      const double stepsizeScale,    
      const double stepsizeExponent,
      const double stepsizeMin,  
      const double stepsizeMax,      
      //const bool   stepsizePrimalDualGapStride,
      //const bool   stepsizeNormalizedSubgradient,
      const SubInfParameter & subInfParam,
      boost::python::tuple  subProbParam
      //const bool useAdaptiveStepsize,     
      //const bool useProjectedAdaptiveStepsize
   ){
      p.decompositionId_=decompositionId;
      p.maximalDualOrder_=maximalDualOrder;
      p.numberOfBlocks_=numberOfBlocks;
      p.maximalNumberOfIterations_=maximalNumberOfIterations;
      p.minimalAbsAccuracy_=minimalAbsAccuracy; 
      p.minimalRelAccuracy_=minimalRelAccuracy;
      p.numberOfThreads_=numberOfThreads;
      p.stepsizeStride_=stepsizeStride;
      p.stepsizeScale_=stepsizeScale;     
      p.stepsizeExponent_=stepsizeExponent;  
      p.stepsizeMin_=stepsizeMin;       
      p.stepsizeMax_=stepsizeMax;      
      p.subPara_=subInfParam;
      //p.stepsizePrimalDualGapStride_=stepsizePrimalDualGapStride;
      //p.stepsizeNormalizedSubgradient_=stepsizeNormalizedSubgradient; 

      SelfType::setSubProbParam(p,subProbParam);

   }





   void static exportInfParam(const std::string & className){
   class_<Parameter > ( className.c_str(),init<>() ) 
      .def_readwrite("decompositionId", & Parameter::decompositionId_,
         "type of decomposition that should be used (independent of model structure) : \n\n"
         "  * 'spanningtrees'\n\n"
         "  * 'trees'\n\n"
         "  * 'blocks'\n\n"
         "  * 'manual' (not yet implemented in python wrapper)\n\n"
      )
      .def_readwrite("maximalDualOrder", & Parameter::maximalDualOrder_,"maximum order of dual variables (order of the corresponding factor)")
      .def_readwrite("numberOfBlocks", & Parameter::numberOfBlocks_,"number of blocks for block decomposition")
      .def_readwrite("maximalNumberOfIterations", & Parameter::maximalNumberOfIterations_,"maximum number of dual iterations")
      .def_readwrite("minimalAbsAccuracy", & Parameter::minimalAbsAccuracy_ ,"the absolut accuracy that has to be guaranteed to stop with an approximate solution (set 0 for optimality)")
      .def_readwrite("minimalRelAccuracy", & Parameter::minimalRelAccuracy_,"the relative accuracy that has to be guaranteed to stop with an approximate solution (set 0 for optimality)")
      .def_readwrite("numberOfThreads", & Parameter::numberOfThreads_,"number of threads for primal problems")
      .def_readwrite("stepsizeStride", & Parameter::stepsizeStride_,"stride stepsize")
      .def_readwrite("stepsizeScale", & Parameter::stepsizeScale_,"scale of the stepsize")
      .def_readwrite("stepsizeExponent", & Parameter::stepsizeExponent_,"stepize exponent")
      .def_readwrite("stepsizeMin", & Parameter::stepsizeMin_,"minimum stepsize")
      .def_readwrite("stepsizeMax", & Parameter::stepsizeMax_,"maximum stepzie")
      //.def_readwrite("stepsizePrimalDualGapStride", & Parameter::stepsizePrimalDualGapStride_,"obsolete")
      //.def_readwrite("stepsizeNormalizedSubgradient", & Parameter::stepsizeNormalizedSubgradient_,"obsolete")
      //Parameter for Subproblems
      .def_readwrite("subInfParam", & Parameter::subPara_,"Sub-Inference parameter")
      .add_property("subProbParam", 
              &SelfType::getSubProbParam, SelfType::setSubProbParam,
      "a tuple with two Bools:\n\n"
      "   - subProbParam[0] is  useAdaptiveStepsize\n\n"    
      "   - subProbParam[1] is  useProjectedAdaptiveStepsize"    
      )
      //.def_readwrite("useAdaptiveStepsize", & Parameter::useAdaptiveStepsize_,"useAdaptiveStepsize")
      //.def_readwrite("useProjectedAdaptiveStepsize", & Parameter::useProjectedAdaptiveStepsize_,"stepsizeNormalizedSubgradient")
      .def ("set", &SelfType::set,
      (
      boost::python::arg("decompositionId")=Parameter::SPANNINGTREES,
      boost::python::arg("maximalDualOrder")=std::numeric_limits<size_t>::max(),
      boost::python::arg("numberOfBlocks")=2,
      boost::python::arg("maximalNumberOfIterations")=100,
      boost::python::arg("minimalAbsAccuracy")=0.0,
      boost::python::arg("minimalRelAccuracy")=0.0,
      boost::python::arg("numberOfThreads")=1,
      boost::python::arg("stepsizeStride")=1,
      boost::python::arg("stepsizeScale")=1,    
      boost::python::arg("stepsizeExponent")=0.5,  
      boost::python::arg("stepsizeMin")=0,  
      boost::python::arg("stepsizeMax")=std::numeric_limits<double>::infinity(),      
      //arg("stepsizePrimalDualGapStride")=false,
      //arg("stepsizeNormalizedSubgradient")=false, 
      boost::python::arg("subInfParam")=SubInfParameter(),
      boost::python::arg("subProbParam")=boost::python::make_tuple(false,false)
      //
      //arg("useAdaptiveStepsize"),
      //arg("useProjectedAdaptiveStepsize")
      ) 
      )
   ; 
   }
};

template<class GM,class SUB_INF,class BLOCK_TYPE>
class InfParamExporter<opengm::DualDecompositionSubGradient<GM,SUB_INF,BLOCK_TYPE> > 
 : public  InfParamExporterDualDecompositionSubGradient<opengm::DualDecompositionSubGradient< GM,SUB_INF,BLOCK_TYPE> > {

};

#endif

