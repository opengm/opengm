#ifndef MULTICUT_CALLER_HXX_
#define MULTICUT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <../include-experimental/opengm/inference/multicut.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MultiCutCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef Multicut<GM, ACC> MultiCut;
   typedef typename MultiCut::VerboseVisitorType VerboseVisitorType;
   typedef typename MultiCut::EmptyVisitorType EmptyVisitorType;
   typedef typename MultiCut::TimingVisitorType TimingVisitorType;
   typename MultiCut::Parameter multicutParameter_;
public:
   const static std::string name_;
   MultiCutCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline MultiCutCaller<IO, GM, ACC>::MultiCutCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of MultiCut caller...", ioIn) {
   addArgument(IntArgument<>(multicutParameter_.numThreads_, "", "threads", "number of threads", multicutParameter_.numThreads_));
   addArgument(BoolArgument(multicutParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(multicutParameter_.cutUp_, "", "cutup", "cut up", multicutParameter_.cutUp_)); 
   addArgument(BoolArgument(multicutParameter_.addNoneFacetDefiningConstraints_,  "", "nfdc", "add non-facet-defining constraints"));
   addArgument(DoubleArgument<>(multicutParameter_.violatedThreshold_, "", "vT", "violation threshold", 0.000001));
   addArgument(DoubleArgument<>(multicutParameter_.timeOut_, "", "timeout", "maximal run-time in seconds", 604800.0)); //default=1week
}

template <class IO, class GM, class ACC>
inline void MultiCutCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running MultiCut caller" << std::endl;

   this-> template infer<MultiCut, TimingVisitorType, typename MultiCut::Parameter>(model, outputfile, verbose, multicutParameter_);
/*   MultiCut multicut(model, multicutParameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(multicut.infer() == NORMAL)) {
      std::string error("MultiCut did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(multicut.arg(states) == NORMAL)) {
      std::string error("MultiCut could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string MultiCutCaller<IO, GM, ACC>::name_ = "MULTICUT";

} // namespace interface

} // namespace opengm

#endif /* MULTICUT_CALLER_HXX_ */
