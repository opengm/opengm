#ifndef LPCPLEX_CALLER_HXX_
#define LPCPLEX_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lpcplex.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPCplexCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef LPCplex<GM, ACC> LPCPLEX;
   typedef typename LPCPLEX::VerboseVisitorType VerboseVisitorType;
   typedef typename LPCPLEX::EmptyVisitorType EmptyVisitorType;
   typedef typename LPCPLEX::TimingVisitorType TimingVisitorType;
   typename LPCPLEX::Parameter lpcplexParameter_;
public:
   const static std::string name_;
   LPCplexCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LPCplexCaller<IO, GM, ACC>::LPCplexCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LPCplex caller...", ioIn) {
   addArgument(BoolArgument(lpcplexParameter_.integerConstraint_, "ic", "integerconstraint", "use integer constraints"));
   addArgument(IntArgument<>(lpcplexParameter_.numberOfThreads_, "", "threads", "number of threads", lpcplexParameter_.numberOfThreads_));
   addArgument(BoolArgument(lpcplexParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(lpcplexParameter_.cutUp_, "", "cutup", "cut up", lpcplexParameter_.cutUp_));
   addArgument(DoubleArgument<>(lpcplexParameter_.timeLimit_,"","timeout","maximal runtime in seconds",604800.0)); //default 1 week
}

template <class IO, class GM, class ACC>
inline void LPCplexCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LPCplex caller" << std::endl;

   this-> template infer<LPCPLEX, TimingVisitorType, typename LPCPLEX::Parameter>(model, outputfile, verbose, lpcplexParameter_);
   /*
   LPCPLEX lpcplex(model, lpcplexParameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(lpcplex.infer() == NORMAL)) {
      std::string error("LPCplex did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(lpcplex.arg(states) == NORMAL)) {
      std::string error("LPCplex could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string LPCplexCaller<IO, GM, ACC>::name_ = "LPCPLEX";

} // namespace interface

} // namespace opengm

#endif /* LPCPLEX_CALLER_HXX_ */
