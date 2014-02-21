#include <iostream>

#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/operations/maximizer.hxx>
#include <opengm/operations/multiplier.hxx>

#include "../cmd_interface.hxx"

//inference caller
#include "../../common/caller/bruteforce_caller.hxx"
#include "../../common/caller/icm_caller.hxx"
#include "../../common/caller/messagepassing_bp_caller.hxx"
#include "../../common/caller/messagepassing_trbp_caller.hxx"
#include "../../common/caller/astar_caller.hxx"
#include "../../common/caller/lazyflipper_caller.hxx"
//#include "../../common/caller/loc_caller.hxx"

#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
#include "../../common/caller/graphcut_caller.hxx"
#include "../../common/caller/alphaexpansion_caller.hxx"
#include "../../common/caller/alphabetaswap_caller.hxx"
#include "../../common/caller/qpbo_caller.hxx"
#endif

#ifdef WITH_CPLEX
#include "../../common/caller/lpcplex_caller.hxx"
#endif


using namespace opengm;

int main(int argc, char** argv) {
   if(argc < 2) {
      std::cerr << "At least one input argument required" << std::endl;
      std::cerr << "try \"-h\" for help" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef Multiplier OperatorType;
   typedef Maximizer AccumulatorType;
   typedef interface::IOCMD InterfaceType;

   // Set functions for graphical model
   typedef meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType>,
      opengm::PottsFunction<ValueType>,
      opengm::PottsNFunction<ValueType>,
      opengm::TruncatedSquaredDifferenceFunction<ValueType>,
      opengm::TruncatedAbsoluteDifferenceFunction<ValueType>
      >::type FunctionTypeList;

   typedef GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList
   > GmType;

   typedef meta::TypeListGenerator <

#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
      interface::GraphCutCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaExpansionCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaBetaSwapCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::QPBOCaller<InterfaceType, GmType, AccumulatorType>,
#endif

#ifdef WITH_CPLEX
      interface::LPCplexCaller<InterfaceType, GmType, AccumulatorType>,
#endif

      interface::ICMCaller<interface::IOCMD, GmType, AccumulatorType>,
      interface::BruteforceCaller<interface::IOCMD, GmType, AccumulatorType>,
      //interface::MessagepassingBPCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::MessagepassingTRBPCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AStarCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::LazyFlipperCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::LOCCaller<InterfaceType, GmType, AccumulatorType>,
      opengm::meta::ListEnd
   >::type InferenceTypeList;

   interface::CMDInterface<GmType, InferenceTypeList> interface(argc, argv);
   interface.parse();

   return 0;
}
