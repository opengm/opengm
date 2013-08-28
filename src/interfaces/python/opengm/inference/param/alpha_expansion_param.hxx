#ifndef ALPHA_EXPANSION_PARAM
#define ALPHA_EXPANSION_PARAM

#include "param_exporter_base.hxx"
//solver specific
#include <opengm/inference/alphaexpansion.hxx>

using namespace boost::python;

template<class INFERENCE>
class InfParamExporterAlphaExpansion{

public:
   typedef typename INFERENCE::ValueType ValueType;
   typedef typename INFERENCE::Parameter Parameter;
   typedef InfParamExporterAlphaExpansion<INFERENCE> SelfType;

   static void set
   (
      Parameter & p,
      const size_t steps,
      const typename Parameter::InferenceParameter & parameter
   ){
      p.maxNumberOfSteps_=steps;
      p.parameter_=parameter;
   }

   void static exportInfParam(const std::string & className){
      class_<Parameter > (className.c_str(), init<>() ) 
         .def(init<const size_t ,const typename Parameter::InferenceParameter & >())
         .def(init<const size_t>())
         .def_readwrite("steps", & Parameter::maxNumberOfSteps_)
         .def_readwrite("subInfParam", & Parameter::parameter_)
         .def ("set", &SelfType::set, 
         ( 
         boost::python::arg("steps")=1000,
         boost::python::arg("subInfParam")=typename Parameter::InferenceParameter()
         ), 
         "Set the parameters values.\n\n"
         "All values of the parameter have a default value.\n\n"
         "Args:\n\n"
         "  steps: Maximum number of iterations (default=1000)\n\n"
         "  graphCutParameter: parameter of the graphcut used within inference (graphCutParameter())\n\n"
         "Returns:\n"
         "  None\n\n"
         ) 
         /*
         ////////// sub gm interface ////
         // - NUMBER OF SUB INFERENCE ARGUMENTS
         .def("_num_sub_inf_param",&selfReturner<Parameter,int>,
            (args("_num_sub_inf_param")=int(1)),"do not call this method with arguments!")
         // - NAME OF THE SUB INFERENCE ARGUMENTS
         .def("_sub_inf_param_0_name",&selfReturner<Parameter,std::string>,
            (args("_sub_inf_param_0_name")=std::string("subInfParam")),"do not call this method with arguments!")
         */
      ; 
   }
};



template<class GM,class ACC,class MIN_ST_CUT>
class InfParamExporter<
      opengm::AlphaExpansion<
         GM,
         opengm::GraphCut<GM,ACC, MIN_ST_CUT> 
      > 
   >  
: public  
   InfParamExporterAlphaExpansion<
      opengm::AlphaExpansion<
         GM,
         opengm::GraphCut<GM,ACC,MIN_ST_CUT> 
      > 
   > {

};


#endif