#ifndef LIBDAI_TREEP_CALLER
#define LIBDAI_TREEP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/tree_expectation_propagation.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiTreeExpectationPropagationCaller : public InferenceCallerBase<IO, GM, ACC, LibDaiTreeExpectationPropagationCaller<IO, GM, ACC> > {
public:
   typedef external::libdai::TreeExpectationPropagation<GM, ACC> InferenceType;
   typedef InferenceCallerBase<IO, GM, ACC, LibDaiTreeExpectationPropagationCaller<IO, GM, ACC> > BaseClass;
   typedef typename InferenceType::VerboseVisitorType VerboseVisitorType;
   typedef typename InferenceType::EmptyVisitorType EmptyVisitorType;
   typedef typename InferenceType::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   LibDaiTreeExpectationPropagationCaller(IO& ioIn);
   virtual ~LibDaiTreeExpectationPropagationCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename InferenceType::Parameter parameter_;
   std::string selectedtreeepType_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::LibDaiTreeExpectationPropagationCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LibDaiTreeExpectationPropagationCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(parameter_.maxiter_, "", "maxIt", "Maximum number of iterations.", size_t(parameter_.maxiter_)));
   addArgument(DoubleArgument<>(parameter_.tolerance_, "", "bound", "convergence bound.", double(parameter_.tolerance_)));
   addArgument(Size_TArgument<>(parameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(parameter_.verbose_)));
   std::vector<std::string> possibleTreeepType;
   possibleTreeepType.push_back(std::string("ORG"));
   possibleTreeepType.push_back(std::string("ALT"));
   addArgument(StringArgument<>(selectedtreeepType_, "", "treepType", "selects the tree-expectation-type", possibleTreeepType.at(0), possibleTreeepType));
}

template <class IO, class GM, class ACC>
inline LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::~LibDaiTreeExpectationPropagationCaller() {

}

template <class IO, class GM, class ACC>
inline void LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
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

   this-> template infer<InferenceType, TimingVisitorType, typename InferenceType::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LibDaiTreeExpectationPropagationCaller<IO, GM, ACC>::name_ = "LIBDAI-TREE-EXPECTATION-PROPAGATION";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_TREEP_CALLER */
