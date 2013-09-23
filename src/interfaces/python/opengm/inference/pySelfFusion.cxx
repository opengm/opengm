#define GraphicalModelDecomposition DualDecompostionSubgradientInference_GraphicalModelDecomposition

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"




#include <opengm/inference/self_fusion.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <param/self_fusion_param.hxx>
#include <param/message_passing_param.hxx>





#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



using namespace boost::python;


template<class GM,class ACC>
void export_self_fusion(){
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   using namespace boost::python;
   import_array();
  



   append_subnamespace("solver");
   
   // documentation 
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "fusion-moves";
   setup.hyperParameterKeyWords        = StringVector(1,std::string("toFuseInf"));
   setup.hyperParametersDoc            = StringVector(1,std::string("inference algorithms to generate labels to fuse"));
   // parameter of inference will change if hyper parameter changes
   setup.hasInterchangeableParameter   = false;



   {



      // set up hyper parameter name for this template
      setup.isDefault=true;
      setup.hyperParameters= StringVector(1,std::string("bp"));

      typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
      typedef opengm::MessagePassing<GM,ACC,UpdateRulesType, opengm::MaxDistance> InfType;
      typedef opengm::SelfFusion<InfType> PySelfFusionInf;


         // export enums
      const std::string enumName1=std::string("_SelfFusion_Bp_FusionSolverType")+srName;
      enum_<typename PySelfFusionInf::FusionSolver> (enumName1.c_str())
         #ifdef WITH_QPBO
         .value("qpbo_fusion",   PySelfFusionInf::QpboFusion)
         #endif
         #ifdef WITH_AD3
         .value("ad3_fusion",  PySelfFusionInf::Ad3Fusion)
         #endif
         .value("astar_fusion",  PySelfFusionInf::AStarFusion)
         .value("lazy_flipper_fusion",  PySelfFusionInf::LazyFlipperFusion)
      ;



      // export parameter
      // exportInfParam<InfType>("_SubParameter_SelfFusion_Bp"); (is exported by bp, same gm)
      exportInfParam<PySelfFusionInf>("_SelfFusion_Bp");
      // export inferences
      class_< PySelfFusionInf>("_SelfFusion_Bp",init<const GM & >())  
      .def(InfSuite<PySelfFusionInf,false>(std::string("SelfFusion"),setup))
      ;
   }


   
}

template void export_self_fusion<opengm::python::GmAdder,opengm::Minimizer>();
