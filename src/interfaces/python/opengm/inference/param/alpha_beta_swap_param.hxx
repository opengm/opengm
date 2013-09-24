#ifndef ALPHA_BETA_SWAP_PARAM
#define ALPHA_BETA_SWAP_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/alphabetaswap.hxx>
#include <opengm/inference/graphcut.hxx>
using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAlphaBetaSwap{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAlphaBetaSwap<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const size_t steps     
   ){
      p.maxNumberOfIterations_=steps;
   }

   void static exportInfParam(const std::string & className){
   class_<Parameter > (className.c_str(), init<>() ) 
         .def_readwrite("steps", & Parameter::maxNumberOfIterations_,
         "steps: Maximum number of iterations"
         )
         .def ("set", &SelfType::set, 
            ( 
            boost::python::arg("steps")=1000
            ), 
         "Set the parameters values.\n\n"
         "All values of the parameter have a default value.\n\n"
         "Args:\n\n"\
         "  steps: Maximum number of iterations (default=1000)\n\n"
         "Returns:\n"
         "  None\n\n"
         ) 
         ; 
      }
};

template<class GM,class ACC,class MIN_ST_CUT>
class InfParamExporter<
      opengm::AlphaBetaSwap<
         GM, opengm::GraphCut< GM,ACC,MIN_ST_CUT> 
      > 
   >  
: public  
   InfParamExporterAlphaBetaSwap<
      opengm::AlphaBetaSwap<
         GM, opengm::GraphCut<GM,ACC,MIN_ST_CUT> 
      > 
   > {

};

#endif