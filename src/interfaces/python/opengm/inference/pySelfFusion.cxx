#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"




#include <opengm/inference/self_fusion.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <opengm/inference/dynamicprogramming.hxx>
#include <opengm/inference/gibbs.hxx>


#include <param/self_fusion_param.hxx>
#include <param/dynamic_programming_param.hxx>
#include <param/dual_decompostion_subgradient_param.hxx>



#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



using namespace boost::python;

template<class INF>
void export_fusion_solver_enums(const std::string & name){

      enum_<typename INF::FusionSolver> (name.c_str())
         #ifdef WITH_QPBO
         .value("qpbo_fusion",         INF::QpboFusion)
         #endif
         #ifdef WITH_AD3
         .value("ad3_fusion",          INF::Ad3Fusion)
         #endif
         #ifdef WITH_CPLEX
         .value("cplex_fusion",          INF::CplexFusion)
         #endif
         .value("astar_fusion",        INF::AStarFusion)
         .value("lazy_flipper_fusion", INF::LazyFlipperFusion)
      ;

}



template<class INF>
void export_all(
   const InfSetup & setup
){
      const std::string extraName = setup.hyperParameters[0];
      const std::string srName = semiRingName  <typename INF::OperatorType,typename INF::AccumulationType >() ;
      // export enums
      const std::string enumName1=std::string("_SelfFusion_")+extraName+("_FusionSolverType")+srName;
      export_fusion_solver_enums<INF>(enumName1);

      const std::string fullName=std::string("_SelfFusion_")+extraName;
      exportInfParam<INF>(fullName);
      // export inferences
      class_< INF >(fullName.c_str(),init<const typename INF::GraphicalModelType & >())  
      .def(InfSuite<INF,false>(std::string("SelfFusion"),setup))
      ;
}


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

   // bp
   {
      typedef opengm::BeliefPropagationUpdateRules<GM,ACC> UpdateRulesType;
      typedef opengm::MessagePassing<GM,ACC,UpdateRulesType, opengm::MaxDistance> InfType;
      typedef opengm::SelfFusion<InfType> PySelfFusionInf;
      // bundled exporter
      setup.isDefault=true;
      setup.hyperParameters= StringVector(1,std::string("bp"));
      export_all<PySelfFusionInf>(setup);
   }
   // trbp
   {
      typedef opengm::TrbpUpdateRules<GM,ACC> UpdateRulesType;
      typedef opengm::MessagePassing<GM,ACC,UpdateRulesType, opengm::MaxDistance> InfType;
      typedef opengm::SelfFusion<InfType> PySelfFusionInf;
      // bundled exporter
      setup.isDefault=false;
      setup.hyperParameters= StringVector(1,std::string("trbp"));
      export_all<PySelfFusionInf>(setup);
   }
   // dd sg
   {
      typedef opengm::DDDualVariableBlock<marray::View<double, false> >                DualBlockType;
      typedef typename opengm::DualDecompositionBase<GM,DualBlockType>::SubGmType      SubGmType;
      typedef opengm::DynamicProgramming<SubGmType, ACC>                               SubInfernce;
      typedef opengm::DualDecompositionSubGradient<GM,SubInfernce,DualBlockType>       InfType;
      typedef opengm::SelfFusion<InfType>                                              PySelfFusionInf;
      // bundled exporter
      setup.isDefault=false;
      setup.hyperParameters= StringVector(1,std::string("dd_sg_dp"));
      export_all<PySelfFusionInf>(setup);
   }
   // gibbs
   {
      typedef opengm::Gibbs<GM,ACC>       InfType;
      typedef opengm::SelfFusion<InfType> PySelfFusionInf;
      // bundled exporter
      setup.isDefault=false;
      setup.hyperParameters= StringVector(1,std::string("gibbs"));
      export_all<PySelfFusionInf>(setup);
   }


   
}

template void export_self_fusion<opengm::python::GmAdder,opengm::Minimizer>();
