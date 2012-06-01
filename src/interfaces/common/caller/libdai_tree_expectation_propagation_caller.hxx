#ifndef LIBDAI_TREEP_CALLER
#define LIBDAI_TREEP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/tree_expectation_propagation.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiTreeExpectationPropagationCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef external::libdai::TreeExpectationPropagation<GM, ACC> InferenceType;
   typedef typename InferenceType::VerboseVisitorType VerboseVisitorType;
   typedef typename InferenceType::EmptyVisitorType EmptyVisitorType;
   typedef typename InferenceType::TimingVisitorType TimingVisitorType;
   typename InferenceType::Parameter parameter_;
   std::string selectedtreeepType_;
public:
   const static std::string name_;
   LibDaiTreeExpectationPropagationCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::LibDaiTreeExpectationPropagationCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LibDaiTreeExpectationPropagationCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(parameter_.maxiter_, "", "maxIt", "Maximum number of iterations.", size_t(parameter_.maxiter_)));
   addArgument(DoubleArgument<>(parameter_.tolerance_, "", "bound", "convergence bound.", double(parameter_.tolerance_)));
   addArgument(Size_TArgument<>(parameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(parameter_.verbose_)));
   std::vector<std::string> possibleTreeepType;
   possibleTreeepType.push_back(std::string("ORG"));
   possibleTreeepType.push_back(std::string("ALT"));
   addArgument(StringArgument<>(selectedtreeepType_, "", "treepType", "selects the tree-expectation-type", possibleTreeepType.at(0), possibleTreeepType));
}

template <class IO, class GM, class ACC>
inline void LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LibDaiTreeExpectationPropagation caller" << std::endl;

   if(selectedtreeepType_ == std::string("ORG")) {
     parameter_.treeEpType_= InferenceType::ORG;
   }
   else if(selectedtreeepType_ == std::string("ALT")) {
     parameter_.treeEpType_= InferenceType::ALT;
   } 
   else {
     throw RuntimeError("Unknown tree-expectation-type for libdai-TreeExpectationPropagation");
   }

   this-> template infer<InferenceType, TimingVisitorType, typename InferenceType::Parameter>(model, outputfile, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::name_ = "LIBDAI-TREE-EXPECTATION-PROPAGATION";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_TREEP_CALLER */
