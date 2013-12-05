//#define GraphicalModelDecomposition DualDecompostionSubgradientInference_GraphicalModelDecomposition

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/dualdecomposition/dualdecomposition_subgradient.hxx>
#include <param/dual_decompostion_subgradient_param.hxx>

// Graph cut subinference
#include <opengm/inference/auxiliary/minstcutboost.hxx>
#ifdef WITH_MAXFLOW
# include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif
#include <opengm/inference/graphcut.hxx>
#include <param/graph_cut_param.hxx>

// Cplex subinference 
#ifdef WITH_CPLEX
#include <opengm/inference/lpcplex.hxx>
#include <param/lpcplex_param.hxx>
#endif

// dynamic programming subinference
#include <param/dynamic_programming_param.hxx>
#include <opengm/inference/dynamicprogramming.hxx>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



using namespace boost::python;


template<class GM,class ACC>
void export_dual_decomposition_subgradient(){

   using namespace boost::python;
   import_array();
  

   typedef typename GM::ValueType                                                   ValueType;
   typedef double                                                                   ViewValueType;
   typedef opengm::DDDualVariableBlock<marray::View<ViewValueType, false> >         DualBlockType;
   typedef typename opengm::DualDecompositionBase<GM,DualBlockType>::SubGmType      SubGmType;
   typedef typename SubGmType::ValueType                                            SubGmValueType;
   // export enums of DualDecompositionBaseParameter
   enum_<opengm::DualDecompositionBaseParameter::DecompositionId> ("DecompositionId")
   .value("manual",        opengm::DualDecompositionBaseParameter::MANUAL)
   .value("tree",          opengm::DualDecompositionBaseParameter::TREE)
   .value("spanningtrees", opengm::DualDecompositionBaseParameter::SPANNINGTREES)
   .value("blocks",        opengm::DualDecompositionBaseParameter::BLOCKS)
   ;
   enum_<opengm::DualDecompositionBaseParameter::DualUpdateId> ("DualUpdateId")
   .value("adaptive",      opengm::DualDecompositionBaseParameter::ADAPTIVE)
   .value("stepsize",      opengm::DualDecompositionBaseParameter::STEPSIZE)
   .value("steplength",    opengm::DualDecompositionBaseParameter::STEPLENGTH)
   .value("kiewil",        opengm::DualDecompositionBaseParameter::KIEWIL)
   ;

   append_subnamespace("solver");
   
   // documentation 
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "dual-decomposition";
   setup.hyperParameterKeyWords        = StringVector(1,std::string("subInference"));
   setup.hyperParametersDoc            = StringVector(1,std::string("inference algorithms for the sub-blocks"));
   // parameter of inference will change if hyper parameter changes
   setup.hasInterchangeableParameter   = false;


   #ifdef WITH_MAXFLOW
   {
      // set up hyper parameter name for this template
      setup.isDefault=false;

      setup.hyperParameters= StringVector(1,std::string("graph-cut"));
      typedef opengm::external::MinSTCutKolmogorov<size_t,SubGmValueType>        MinStCutKolmogorov;
      typedef opengm::GraphCut<SubGmType, ACC, MinStCutKolmogorov>               SubInfernce;
      typedef opengm::DualDecompositionSubGradient<GM,SubInfernce,DualBlockType> PyDualDecomposition;
      // export parameter
      exportInfParam<SubInfernce>("_SubParameter_DualDecompositionSubgradient_GraphCutKolmogorov");
      exportInfParam<PyDualDecomposition>("_DualDecompositionSubgradient_GraphCutKolmogorov");
      // export inferences
      class_< PyDualDecomposition>("_DualDecompositionSubgradient_GraphCutKolmogorov",init<const GM & >())  
      .def(InfSuite<PyDualDecomposition,false>(std::string("DualDecompositionSubgradient"),setup))
      ;
   }
   #else
   {
      // set up hyper parameter name for this template
      setup.isDefault=false;

      setup.hyperParameters= StringVector(1,std::string("graph-cut"));
      typedef opengm::MinSTCutBoost<size_t, ValueType, opengm::KOLMOGOROV>       MinStCutBoostKolmogorov;
      typedef opengm::GraphCut<SubGmType,ACC, MinStCutBoostKolmogorov>           SubInfernce;
      typedef opengm::DualDecompositionSubGradient<GM,SubInfernce,DualBlockType> PyDualDecomposition;
      // export parameter
      exportInfParam<SubInfernce>("_SubParameter_DualDecompositionSubgradient_GraphCutBoostKolmogorov");
      exportInfParam<PyDualDecomposition>("_DualDecompositionSubgradient_GraphCutBoostKolmogorov");
      // export inferences
      class_< PyDualDecomposition>("_DualDecompositionSubgradient_GraphCutBoostKolmogorov",init<const GM & >())  
      .def(InfSuite<PyDualDecomposition,false>(std::string("DualDecompositionSubgradient"),setup))
      ;
   }
   #endif

   /*
   #ifdef WITH_CPLEX
   {
      // set up hyper parameter name for this template
      setup.isDefault=false;

      setup.hyperParameters= StringVector(1,std::string("cplex"));
      typedef opengm::LPCplex<SubGmType, ACC>                                    SubInfernce;
      typedef opengm::DualDecompositionSubGradient<GM,SubInfernce,DualBlockType> PyDualDecomposition;

      exportInfParam<SubInfernce>("_SubParameter_DualDecompositionSubgradient_Cplex");
      exportInfParam<PyDualDecomposition>("DualDecompositionSubgradient_Cplex");
      // export inferences
      class_< PyDualDecomposition>("_DualDecompositionSubgradient_Cplex",init<const GM & >())  
      .def(InfSuite<PyDualDecomposition,false>(std::string("DualDecompositionSubgradient"),setup))
      ;
 
   }
   #endif
   */

   {

      // set up hyper parameter name for this template
      setup.isDefault = true;

      setup.hyperParameters= StringVector(1,std::string("dynamic-programming"));
      typedef opengm::DynamicProgramming<SubGmType, ACC>                         SubInfernce;
      typedef opengm::DualDecompositionSubGradient<GM,SubInfernce,DualBlockType> PyDualDecomposition;

      exportInfParam<SubInfernce>("_SubParameter_DualDecompositionSubgradient_DynamicProgramming");
      exportInfParam<PyDualDecomposition>("_DualDecompositionSubgradient_DynamicProgramming");
      // export inferences
      class_< PyDualDecomposition>("_DualDecompositionSubgradient_DynamicProgramming",init<const GM & >())  
      .def(InfSuite<PyDualDecomposition,false>(std::string("DualDecompositionSubgradient"),setup))
      ;
 
   }

   
}

template void export_dual_decomposition_subgradient<opengm::python::GmAdder,opengm::Minimizer>();
