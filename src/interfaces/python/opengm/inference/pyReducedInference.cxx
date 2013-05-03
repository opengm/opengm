
// redefined symbol since pyTrws.cxx:(.text+0x0): multiple definition of `DefaultErrorFn(char*)'
#define DefaultErrorFn DefaultErrorFn_ReducedInference 


#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/reducedinference.hxx>
#include <param/reduced_inference_param.hxx>

// SUBSOLVERS
//#ifdef WITH_MULTICUT
//#include <opengm/inference/multicut.hxx>
//#endif
#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#endif
//#ifdef WITH_FASTPD
//#include <opengm/inference/external/fastPD.hxx>
//#endif
#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#include <param/trws_external_param.hxx>
#endif
//#ifdef WITH_GCOLIB
//#include <opengm/inference/external/gco.hxx>
//#endif


using namespace boost::python;


template<class GM,class ACC>
void export_reduced_inference(){

   using namespace boost::python;
   import_array();
  
   typedef opengm::ReducedInferenceHelper<GM> RedInfHelper;
   typedef typename RedInfHelper::InfGmType SubGmType;

   

   append_subnamespace("solver");
   
   // documentation 
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "qpbo-reduced inference";
   setup.hyperParameterKeyWords        = StringVector(1,std::string("subInference"));
   setup.hyperParametersDoc            = StringVector(1,std::string("inference algorithms for the sub-problems"));
   // parameter of inference will change if hyper parameter changes
   setup.hasInterchangeableParameter   = false;



   {
      #ifdef WITH_TRWS

      typedef opengm::external::TRWS<SubGmType> SubInfType;
      typedef opengm::ReducedInference<GM,ACC,SubInfType> PyReducedInf;

      // set up hyper parameter name for this template
      setup.isDefault = true;
      setup.hyperParameters= StringVector(1,std::string("trws"));

      // THE ENUMS OF SUBSOLVERS NEED TO BE EXPORTED AGAIN.... THIS COULD BE SOLVED BETTER....
      const std::string enumName=std::string("_SubInference_ReducedInference_Trws_EnergyType")+srName;
      enum_<typename SubInfType::Parameter::EnergyType> (enumName.c_str())
         .value("view", SubInfType::Parameter::VIEW)
         .value("tables", SubInfType::Parameter::TABLES)
         .value("tl1", SubInfType::Parameter::TL1)
         .value("tl2", SubInfType::Parameter::TL2)
      ;
         
      // export sub-inf param and param
      exportInfParam<exportTag::NoSubInf,SubInfType>("_SubParameter_ReducedInference_Trws");
      exportInfParam<exportTag::NoSubInf,PyReducedInf>("_ReducedInference_Trws");
      // export inference itself
      class_< PyReducedInf>("_ReducedInference_Trws",init<const GM & >())  
      .def(InfSuite<PyReducedInf,false>(std::string("ReducedInference"),setup))
      ;

      #endif 
   }

   
}

template void export_reduced_inference<GmAdder,opengm::Minimizer>();
