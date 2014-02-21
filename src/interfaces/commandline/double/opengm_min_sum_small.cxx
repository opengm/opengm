#include <iostream>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/minimizer.hxx>
#include <opengm/operations/adder.hxx>
#include <opengm/functions/explicit_function.hxx>
#include <opengm/functions/potts.hxx>
#include <opengm/functions/pottsn.hxx>
#include <opengm/functions/pottsg.hxx>
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "../cmd_interface.hxx"

//inference caller
#include "../../common/caller/bruteforce_caller.hxx"
#include "../../common/caller/icm_caller.hxx"
#include "../../common/caller/messagepassing_bp_caller.hxx"
#include "../../common/caller/messagepassing_trbp_caller.hxx"
#include "../../common/caller/astar_caller.hxx"
#include "../../common/caller/lazyflipper_caller.hxx"
//#include "../../common/caller/gibbs_caller.hxx"
//#include "../../common/caller/swendsenwang_caller.hxx"

#ifdef WITH_TRWS
#include "../../common/caller/trws_caller.hxx"
#endif

#ifdef WITH_AD3
#include "../../common/caller/loc_caller.hxx"
#endif

#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
#include "../../common/caller/graphcut_caller.hxx"
#include "../../common/caller/alphaexpansion_caller.hxx"
#include "../../common/caller/alphabetaswap_caller.hxx"
#include "../../common/caller/qpbo_caller.hxx"
#endif

#ifdef WITH_CPLEX
#include "../../common/caller/lpcplex_caller.hxx"
#endif

#ifdef WITH_BOOST
#include "../../common/caller/sat_caller.hxx"
#endif

#ifdef WITH_BUNDLE
#include "../../common/caller/dd_bundle_caller.hxx"
#endif
#include "../../common/caller/dd_subgradient_caller.hxx"

#ifdef WITH_MRF
#include "../../common/caller/mrflib_caller.hxx"
#endif

using namespace opengm;

int main(int argc, char** argv) {
   if(argc < 2) {
      std::cerr << "At least one input argument required" << std::endl;
      std::cerr << "try \"-h\" for help" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef size_t IndexType;
   typedef size_t LabelType;
   typedef Adder OperatorType;
   typedef Minimizer AccumulatorType;
   typedef interface::IOCMD InterfaceType;
   typedef DiscreteSpace<IndexType, LabelType> SpaceType;

   // Set functions for graphical model

   typedef meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType,IndexType,LabelType>,
      opengm::PottsFunction<ValueType,IndexType,LabelType>,
      opengm::PottsNFunction<ValueType,IndexType,LabelType>,
      opengm::PottsGFunction<ValueType,IndexType,LabelType>,
      opengm::TruncatedSquaredDifferenceFunction<ValueType,IndexType,LabelType>,
      opengm::TruncatedAbsoluteDifferenceFunction<ValueType,IndexType,LabelType> 
      >::type FunctionTypeList;


   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList,
      SpaceType
   > GmType;

   typedef meta::TypeListGenerator < 
      interface::ICMCaller<interface::IOCMD, GmType, AccumulatorType>,
      interface::BruteforceCaller<interface::IOCMD, GmType, AccumulatorType>,
      //interface::MessagepassingBPCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::MessagepassingTRBPCaller<InterfaceType, GmType, AccumulatorType>,//
      //     interface::AStarCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::LazyFlipperCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::GibbsCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::SwendsenWangCaller<InterfaceType, GmType, AccumulatorType>,
      opengm::meta::ListEnd
   >::type NativeInferenceTypeList;

   typedef meta::TypeListGenerator <
#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
      interface::GraphCutCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaExpansionCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaBetaSwapCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::QPBOCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_AD3
      //interface::LOCCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_CPLEX
      interface::LPCplexCaller<InterfaceType, GmType, AccumulatorType>,
#endif

#ifdef WITH_DD
#ifdef WITH_BUNDLE
      interface::DDBundleCaller<InterfaceType, GmType, AccumulatorType>,
#endif
      interface::DDSubgradientCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_TRWS
      interface::TRWSCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_MRF
      interface::MRFLIBCaller<InterfaceType, GmType, AccumulatorType>,
#endif
      opengm::meta::ListEnd
   >::type ExternalInferenceTypeList;

   typedef meta::MergeTypeLists<NativeInferenceTypeList, ExternalInferenceTypeList>::type InferenceTypeList;
   interface::CMDInterface<GmType, InferenceTypeList> interface(argc, argv);
   interface.parse();

   return 0;
}
