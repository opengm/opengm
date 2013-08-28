#ifdef WITH_LIBDAI
#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"



#include <opengm/inference/external/libdai/bp.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>
#include <opengm/inference/external/libdai/junction_tree.hxx>
#include <opengm/inference/external/libdai/gibbs.hxx>
#include <opengm/inference/external/libdai/dec_map.hxx>
#include <param/libdai_param.hxx>

using namespace boost::python;







template<class GM,class ACC>
void export_libdai_inference(){




	using namespace boost::python;
   import_array();
   append_subnamespace("solver");
   
   // INFERENCE TYPEDEFS
	typedef GM PyGm;
	typedef typename PyGm::ValueType ValueType;
	typedef typename PyGm::IndexType IndexType;
	typedef typename PyGm::LabelType LabelType;
   // bp
	typedef opengm::external::libdai::Bp<PyGm, ACC>                      PyLibdaiBp;  
   //fractional bp
	typedef opengm::external::libdai::FractionalBp<PyGm, ACC>            PyLibdaiFractionalBp;
   // trbp
   typedef opengm::external::libdai::TreeReweightedBp<PyGm, ACC>        PyLibdaiTrbp;
   // junction tree
   typedef opengm::external::libdai::JunctionTree<PyGm, ACC>            PyLibdaiJunctionTree;
   // gibbs
   typedef opengm::external::libdai::Gibbs<PyGm, ACC>                   PyLibdaiGibbs;

   // dec map
   typedef opengm::external::libdai::DecMap<PyLibdaiBp>                 PyLibDaiDecMapBp;
   typedef opengm::external::libdai::DecMap<PyLibdaiBp>                 PyLibDaiDecMapFractionalBp;
   typedef opengm::external::libdai::DecMap<PyLibdaiBp>                 PyLibDaiDecMapTrwBp;
   //
   //typedef opengm::external::libdai::DoubleLoopGeneralizedBP<PyGm, ACC> PyLibDaiDoubleLoopGeneralizedBP;

   // setup 
   InfSetup setup;
   std::string srName = semiRingName  <typename GM::OperatorType,ACC >() ;
   setup.dependencies = "This algorithm needs LibDai, compile OpenGM with CMake-Flag ``WITH_LIBDAI`` set to ``ON`` ";

   // bp
   {
      setup.algType    = "message-passing";
      setup.guarantees = "";
      typedef PyLibdaiBp LibDaiInf;
      const std::string enumName=std::string("_BpUpdateRuleLibDai")+srName;
      enum_<typename LibDaiInf::UpdateRule> (enumName.c_str())
      .value("parall", LibDaiInf::PARALL)
      .value("seqfix", LibDaiInf::SEQFIX)
      .value("seqrnd", LibDaiInf::SEQRND)
      .value("seqmax", LibDaiInf::SEQMAX)
      ;
      
      // export parameter
      exportInfParam<LibDaiInf>("_BeliefPropagationLibDai");
      // export inference
      class_< LibDaiInf>("_BeliefPropagationLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("BeliefPropagationLibDai"),setup))
      ;
   }
   // fractional bp
   {
      setup.algType    = "message-passing";
      setup.guarantees = "";
      typedef PyLibdaiFractionalBp LibDaiInf;
      const std::string enumName=std::string("_FractionalBpUpdateRuleLibDai")+srName;
      enum_<typename LibDaiInf::UpdateRule> (enumName.c_str())
      .value("parall", LibDaiInf::PARALL)
      .value("seqfix", LibDaiInf::SEQFIX)
      .value("seqrnd", LibDaiInf::SEQRND)
      .value("seqmax", LibDaiInf::SEQMAX)
      ;
      
      // export parameter
      exportInfParam<LibDaiInf>("_FractionalBpLibDai");
      // export inference
      class_< LibDaiInf>("_FractionalBpLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("FractionalBpLibDai"),setup))
      ;
   }
   // trw bp
   {
      setup.algType    = "message-passing";
      setup.guarantees = "";
      typedef PyLibdaiTrbp LibDaiInf;
      const std::string enumName=std::string("_TreeReweightedBpUpdateRuleLibDai")+srName;
      enum_<typename LibDaiInf::UpdateRule> (enumName.c_str())
      .value("parall", LibDaiInf::PARALL)
      .value("seqfix", LibDaiInf::SEQFIX)
      .value("seqrnd", LibDaiInf::SEQRND)
      .value("seqmax", LibDaiInf::SEQMAX)
      ;
      
      // export parameter
      exportInfParam<LibDaiInf>("_TreeReweightedBpLibDai");
      // export inference
      class_< LibDaiInf>("_TreeReweightedBpLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("TreeReweightedBpLibDai"),setup))
      ;
   }
   // junction tree
   {
      setup.algType    = "dynamic-programming";
      setup.guarantees = "global optimal";
      typedef PyLibdaiJunctionTree LibDaiInf;

      const std::string enumName1=std::string("_JunctionTreeUpdateRuleLibDai")+srName;
      enum_<typename LibDaiInf::UpdateRule> (enumName1.c_str())
      .value("hugin", LibDaiInf::HUGIN)
      .value("shsh", LibDaiInf::SHSH)
      ;
      const std::string enumName2=std::string("_JunctionTreeHeuristicLibDai" )+srName;
      enum_<typename LibDaiInf::Heuristic> (enumName2.c_str())
      .value("minfill", LibDaiInf::MINFILL)
      .value("weightedminfill", LibDaiInf::WEIGHTEDMINFILL)
      .value("minweight", LibDaiInf::MINWEIGHT)
      .value("minneighbors", LibDaiInf::MINNEIGHBORS)
      ;

      // export parameter
      exportInfParam<LibDaiInf>("_JunctionTreeLibDai");
      // export inference
      class_< LibDaiInf>("_JunctionTreeLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("JunctionTreeLibDai"),setup))
      ;
   }
   // gibbs
   {
      setup.algType    = "sampling";
      setup.guarantees = "";
      typedef PyLibdaiGibbs LibDaiInf;

      // export parameter
      exportInfParam<LibDaiInf>("_GibbsLibDai");
      // export inference
      class_< LibDaiInf>("GibbsLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("GibbsLibDai"),setup))
      ;
   }
   // decmap
   {  
      setup.algType    ="decimation-algorithm";
      setup.cite       = "";
      setup.guarantees = "";
      setup.notes      ="Approximate inference algorithm DecMAP, which constructs a MAP state by decimation.\n\n"
                        "Decimation involves repeating the following two steps until no free variables remain:\n\n"
                        "Run an approximate inference algorithm,clamp the factor with the lowest entropy to its most probable state\n\n";
      setup.hyperParameterKeyWords        = StringVector(1,std::string("subInference"));
      setup.hyperParametersDoc            = StringVector(1,std::string("sub-inference algorithms of dec-map"));
      // parameter of inference will change if hyper parameter changes
      setup.hasInterchangeableParameter   = false;
      // decmap-bp
      {
         typedef opengm::external::libdai::DecMap<PyLibdaiBp>  LibDaiInf;
         setup.isDefault=true;
         setup.hyperParameters= StringVector(1,std::string("bp"));
         // export parameter
         exportInfParam<LibDaiInf>("_DecimationBpLibDai");
         // export inference
         class_< LibDaiInf>("_DecimationBpLibDai",init<const GM & >())  
         .def(InfSuite<LibDaiInf,false>(std::string("DecimationLibDai"),setup))
         ;
      }
      // decmap-fractionalBp
      {
         typedef opengm::external::libdai::DecMap<PyLibdaiFractionalBp>  LibDaiInf;
         setup.isDefault=false;
         setup.hyperParameters= StringVector(1,std::string("fractionalBp"));
         // export parameter
         exportInfParam<LibDaiInf>("_DecimationFractionalBpLibDai");
         // export inference
         class_< LibDaiInf>("_DecimationFractionalBpLibDai",init<const GM & >())  
         .def(InfSuite<LibDaiInf,false>(std::string("DecimationLibDai"),setup))
         ;
      }

      // decmap-trwbp
      {
         typedef opengm::external::libdai::DecMap<PyLibdaiTrbp>  LibDaiInf;
         setup.isDefault=false;
         setup.hyperParameters= StringVector(1,std::string("trwBp"));
         // export parameter
         exportInfParam<LibDaiInf>("_DecimationTrwBpLibDai");
         // export inference
         class_< LibDaiInf>("_DecimationTrwBpLibDai",init<const GM & >())  
         .def(InfSuite<LibDaiInf,false>(std::string("DecimationLibDai"),setup))
         ;
      }
      // decmap-trwbp
      {
         typedef opengm::external::libdai::DecMap<PyLibdaiGibbs>  LibDaiInf;
         setup.isDefault=false;
         setup.hyperParameters= StringVector(1,std::string("gibbs"));
         // export parameter
         exportInfParam<LibDaiInf>("_DecimationGibbsLibDai");
         // export inference
         class_< LibDaiInf>("_DecimationGibbsLibDai",init<const GM & >())  
         .def(InfSuite<LibDaiInf,false>(std::string("DecimationLibDai"),setup))
         ;
      }
   }

   // throws Feature not implemented Feature not implemented
   /*
   // double loop generalized bp
   {
      setup.algType    = "message-passing";
      setup.guarantees = "";
      typedef PyLibDaiDoubleLoopGeneralizedBP LibDaiInf;

      const std::string enumName1=std::string("_DoubleLoopGenereralizedBpInitLibDai")+srName;
      enum_<typename LibDaiInf::Init> (enumName1.c_str())
         .value("uniform", LibDaiInf::UNIFORM)
         .value("random", LibDaiInf::RANDOM)
      ;
      const std::string enumName2=std::string("_DoubleLoopGenereralizedBpClustersLibDai")+srName;
      enum_<typename LibDaiInf::Clusters> (enumName2.c_str())
         .value("min", LibDaiInf::MIN)
         .value("bethe", LibDaiInf::BETHE)
         .value("delta", LibDaiInf::DELTA)
         .value("loop", LibDaiInf::LOOP)
      ;    

      // export parameter
      exportInfParam<LibDaiInf>("_DoubleLoopGenereralizedBpLibDai");
      // export inference
      class_< LibDaiInf>("DoubleLoopGenereralizedBpLibDai",init<const GM & >())  
      .def(InfSuite<LibDaiInf,false>(std::string("DoubleLoopGenereralizedBpLibDai"),setup))
      ;
   }
   */
}

template void export_libdai_inference<opengm::python::GmAdder, opengm::Minimizer>();
//template void export_libdai_inference<GmAdder, opengm::Maximizer>();
//template void export_libdai_inference<GmMultiplier, opengm::Minimizer>();
//template void export_libdai_inference<GmMultiplier, opengm::Maximizer>();

#endif
