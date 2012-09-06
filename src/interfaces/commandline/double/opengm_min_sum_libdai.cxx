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
#ifdef WITH_LIBDAI
#include "../../common/caller/libdai_bp_caller.hxx"
#include "../../common/caller/libdai_trwbp_caller.hxx"
#include "../../common/caller/libdai_junctiontree_caller.hxx"
#include "../../common/caller/libdai_double_loop_generalized_bp_caller.hxx"
#include "../../common/caller/libdai_tree_expectation_propagation_caller.hxx"
#include "../../common/caller/libdai_fractional_bp_caller.hxx"
#endif

using namespace opengm;

int main(int argc, char** argv) {
   if(argc < 2) {
      std::cerr << "At least one input argument required" << std::endl;
      std::cerr << "try \"-h\" for help" << std::endl;
      return 1;
   }

   typedef double ValueType;
   typedef Adder OperatorType;
   typedef Minimizer AccumulatorType;
   typedef interface::IOCMD InterfaceType;

   // Set functions for graphical model

   typedef meta::TypeListGenerator<
      opengm::ExplicitFunction<ValueType>,
      opengm::PottsFunction<ValueType>,
      opengm::PottsNFunction<ValueType>,
      opengm::PottsGFunction<ValueType>,
      opengm::TruncatedSquaredDifferenceFunction<ValueType>,
      opengm::TruncatedAbsoluteDifferenceFunction<ValueType> 
      >::type FunctionTypeList;


   typedef opengm::GraphicalModel<
      ValueType,
      OperatorType,
      FunctionTypeList
   > GmType;

   typedef meta::TypeListGenerator <
   
#ifdef WITH_LIBDAI
      interface::LibDaiBpCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LibDaiTrwBpCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LibDaiFractionalBpCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LibDaiJunctionTreeCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LibDaiDoubleLoopGeneralizedBpCaller<InterfaceType, GmType, AccumulatorType>,
      interface::LibDaiTreeExpectationPropagationCaller<InterfaceType, GmType, AccumulatorType>,
#endif
      opengm::meta::ListEnd
   >::type InferenceTypeList;
   
   interface::CMDInterface<GmType, InferenceTypeList> interface(argc, argv);
   interface.parse();

   return 0;
}
