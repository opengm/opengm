//include matlab headers
#include "mex.h"

// matlab handle
#include "../helper/handle/handle.hxx"

#include "../model/matlabModelType.hxx"
#include "matlab_interface.hxx"

//inference caller
#include <../src/interfaces/common/caller/bruteforce_caller.hxx>
#include <../src/interfaces/common/caller/icm_caller.hxx>
#include <../src/interfaces/common/caller/messagepassing_bp_caller.hxx>
#include <../src/interfaces/common/caller/messagepassing_trbp_caller.hxx>
#include <../src/interfaces/common/caller/astar_caller.hxx>
#include <../src/interfaces/common/caller/lazyflipper_caller.hxx>
//#include <../src/interfaces/common/caller/gibbs_caller.hxx>
//#include <../src/interfaces/common/caller/swendsenwang_caller.hxx>

#ifdef WITH_TRWS
#include <../src/interfaces/common/caller/trws_caller.hxx>
#endif

#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
#include <../src/interfaces/common/caller/graphcut_caller.hxx>
#include <../src/interfaces/common/caller/alphaexpansion_caller.hxx>
#include <../src/interfaces/common/caller/alphabetaswap_caller.hxx>
#include <../src/interfaces/common/caller/qpbo_caller.hxx>
#endif

#ifdef WITH_CPLEX
#include <../src/interfaces/common/caller/multicut_caller.hxx>
#include <../src/interfaces/common/caller/lpcplex_caller.hxx>
#endif

#ifdef WITH_BOOST
#include <../src/interfaces/common/caller/sat_caller.hxx>
#endif

#ifdef WITH_DD
#ifdef WITH_BUNDLE
#include <../src/interfaces/common/caller/dd_bundle_caller.hxx>
#endif
#include <../src/interfaces/common/caller/dd_subgradient_caller.hxx>
#endif

#ifdef WITH_MRF
#include <../src/interfaces/common/caller/mrflib_caller.hxx>
#endif

#ifdef WITH_GCO
#include <../src/interfaces/common/caller/gcolib_caller.hxx>
#endif

#ifdef WITH_FASTPD
#include <../src/interfaces/common/caller/fastPD_caller.hxx>
#endif

#ifdef WITH_GRANTE
#include <../src/interfaces/common/caller/grante_caller.hxx>
#endif

#ifdef WITH_AD3
#include <../src/interfaces/common/caller/loc_caller.hxx>
#endif

using namespace opengm;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
   if(nrhs < 1) {
      mexErrMsgTxt("At least one input argument required \ntry 'h' for help");
   }
   typedef interface::IOMatlab InterfaceType;
   typedef opengm::Minimizer AccumulatorType;
   typedef opengm::interface::MatlabModelType::GmType GmType;

   typedef meta::TypeListGenerator <
      interface::ICMCaller<InterfaceType, GmType, AccumulatorType>,
      interface::BruteforceCaller<InterfaceType, GmType, AccumulatorType>,
      interface::MessagepassingBPCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::MessagepassingTRBPCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AStarCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LazyFlipperCaller<InterfaceType, GmType, AccumulatorType>
      //interface::GibbsCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::SwendsenWangCaller<InterfaceType, GmType, AccumulatorType>
      >::type NativeInferenceTypeList;

   typedef meta::TypeListGenerator <
#if defined(WITH_MAXFLOW) || defined(WITH_BOOST)
      interface::GraphCutCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaExpansionCaller<InterfaceType, GmType, AccumulatorType>,
      interface::AlphaBetaSwapCaller<InterfaceType, GmType, AccumulatorType>,
      //interface::QPBOCaller<InterfaceType, GmType, AccumulatorType>,
#endif

#ifdef WITH_CPLEX
      interface::MultiCutCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LPCplexCaller<InterfaceType, GmType, AccumulatorType>,
#endif

#ifdef WITH_DD
#ifdef WITH_BUNDLE
      //interface::DDBundleCaller<InterfaceType, GmType, AccumulatorType>,
#endif
      //interface::DDSubgradientCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_TRWS
      interface::TRWSCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_MRF
      interface::MRFLIBCaller<InterfaceType, GmType, AccumulatorType>,
#endif
/*#ifdef WITH_GCO
      interface::GCOLIBCaller<InterfaceType, GmType, AccumulatorType>,
#endif*/
/*#ifdef WITH_FASTPD
      interface::FastPDCaller<InterfaceType, GmType, AccumulatorType>,
#endif*/
#ifdef WITH_GRANTE
      //interface::GranteCaller<InterfaceType, GmType, AccumulatorType>,
#endif
#ifdef WITH_AD3
      //interface::LOCCaller<InterfaceType, GmType, AccumulatorType>,
#endif
      opengm::meta::ListEnd
      >::type ExternalInferenceTypeList;

   typedef meta::MergeTypeLists<NativeInferenceTypeList, ExternalInferenceTypeList>::type InferenceTypeList;
   interface::MatlabInterface<GmType, InferenceTypeList> interface(nlhs, plhs, nrhs, prhs);
   interface.parse();
}
