#ifdef WITH_CONICBUNDLE
//#define GraphicalModelDecomposition DualDecompostionBundleInference_GraphicalModelDecomposition

#include <boost/python.hpp>
#include <string>
#include "inf_def_visitor.hxx"

#include <opengm/inference/dualdecomposition/dualdecomposition_bundle.hxx>
#include <param/dual_decompostion_bundle_param.hxx>

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
void export_dual_decomposition_bundle(){

   using namespace boost::python;
   import_array();
   
   typedef typename GM::ValueType                                                   ValueType;
   typedef double                                                                   ViewValueType;
   typedef opengm::DDDualVariableBlock2<marray::View<ValueType,false> >             DualBlockType;
   typedef typename opengm::DualDecompositionBase<GM,DualBlockType>::SubGmType      SubGmType;
   typedef typename SubGmType::ValueType                                            SubGmValueType;

   append_subnamespace("solver");
   
   // documentation 
   InfSetup setup;
   setup.cite       = "";
   setup.algType    = "dual-decomposition";
   setup.hyperParameterKeyWords        = StringVector(1,std::string("subInference"));
   setup.hyperParametersDoc            = StringVector(1,std::string("inference algorithms for the sub-blocks"));
   // parameter of inference will change if hyper parameter changes
   setup.hasInterchangeableParameter   = false;

   {
      // set up hyper parameter name for this template
      setup.isDefault = true;

      setup.hyperParameters= StringVector(1,std::string("dynamic-programming"));
      typedef opengm::DynamicProgramming<SubGmType, ACC> SubInfernce;
      typedef opengm::DualDecompositionBundle<GM,SubInfernce,DualBlockType> PyDualDecomposition;

      exportInfParam<SubInfernce>("_SubParameter_DualDecompositionBundle_DynamicProgramming");
      exportInfParam<PyDualDecomposition>("_DualDecompositionBundle_DynamicProgramming");
      // export inferences
      class_< PyDualDecomposition>("_DualDecompositionBundle_DynamicProgramming",init<const GM & >())  
      .def(InfSuite<PyDualDecomposition,false>(std::string("DualDecompositionBundle"),setup))
      ;
 
   }

   
}

template void export_dual_decomposition_bundle<GmAdder,opengm::Minimizer>();

#endif // WITH_CONICBUNDLE