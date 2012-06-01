#ifndef LIBDAI_TRW_BP_CALLER
#define LIBDAI_TRW_BP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/tree_reweighted_bp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiTrwBpCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef external::libdai::TreeReweightedBp<GM, ACC> LibDai_TrwBp;
   typedef typename LibDai_TrwBp::VerboseVisitorType VerboseVisitorType;
   typedef typename LibDai_TrwBp::EmptyVisitorType EmptyVisitorType;
   typedef typename LibDai_TrwBp::TimingVisitorType TimingVisitorType;
   typename LibDai_TrwBp::Parameter trwbpParameter_;
   std::string selectedUpdateRule_;
public:
   const static std::string name_;
   LibDaiTrwBpCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LibDaiTrwBpCaller<IO, GM, ACC>::LibDaiTrwBpCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LibDaiTrwBpCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(trwbpParameter_.maxIterations_, "", "maxIt", "Maximum number of iterations.", size_t(trwbpParameter_.maxIterations_)));
   addArgument(DoubleArgument<>(trwbpParameter_.tolerance_, "", "bound", "convergence bound.", double(trwbpParameter_.tolerance_)));
   addArgument(DoubleArgument<>(trwbpParameter_.damping_, "", "damping", "message damping", double(0.0)));
   addArgument(Size_TArgument<>(trwbpParameter_.ntrees_, "", "nTrees", "number of trees", size_t(0)));
   addArgument(Size_TArgument<>(trwbpParameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(trwbpParameter_.verbose_)));
   std::vector<std::string> possibleUpdateRule;
   possibleUpdateRule.push_back(std::string("PARALL"));
   possibleUpdateRule.push_back(std::string("SEQFIX"));
   possibleUpdateRule.push_back(std::string("SEQRND"));
   possibleUpdateRule.push_back(std::string("SEQMAX"));
   addArgument(StringArgument<>(selectedUpdateRule_, "", "updateRule", "selects the update rule", possibleUpdateRule.at(0), possibleUpdateRule));
}

template <class IO, class GM, class ACC>
inline void LibDaiTrwBpCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LibDaiTrwBp caller" << std::endl;

   if(selectedUpdateRule_ == std::string("PARALL")) {
     trwbpParameter_.updateRule_= LibDai_TrwBp::PARALL;
   }
   else if(selectedUpdateRule_ == std::string("SEQFIX")) {
     trwbpParameter_.updateRule_= LibDai_TrwBp::SEQFIX;
   } 
   else if(selectedUpdateRule_ == std::string("SEQMAX")) {
     trwbpParameter_.updateRule_= LibDai_TrwBp::SEQMAX;
   }
   else if(selectedUpdateRule_ == std::string("SEQRND")) {
     trwbpParameter_.updateRule_= LibDai_TrwBp::SEQRND;
   }
   else {
     throw RuntimeError("Unknown update rule for libdai-trw-bp");
   }

   this-> template infer<LibDai_TrwBp, TimingVisitorType, typename LibDai_TrwBp::Parameter>(model, outputfile, verbose, trwbpParameter_);

}

template <class IO, class GM, class ACC>
const std::string LibDaiTrwBpCaller<IO, GM, ACC>::name_ = "LIBDAI-TRW-BP";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_TRW_BP_CALLER */
