#ifndef LPCPLEX_CALLER_HXX_
#define LPCPLEX_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lpcplex.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPCplexCaller : public InferenceCallerBase<IO, GM, ACC, LPCplexCaller<IO, GM, ACC> > {
public:
   typedef LPCplex<GM, ACC> LPCPLEX;
   typedef InferenceCallerBase<IO, GM, ACC, LPCplexCaller<IO, GM, ACC> > BaseClass;
   typedef typename LPCPLEX::VerboseVisitorType VerboseVisitorType;
   typedef typename LPCPLEX::EmptyVisitorType EmptyVisitorType;
   typedef typename LPCPLEX::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LPCplexCaller(IO& ioIn);
   virtual ~LPCplexCaller();
protected:

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename LPCPLEX::Parameter lpcplexParameter_;

};

template <class IO, class GM, class ACC>
inline LPCplexCaller<IO, GM, ACC>::LPCplexCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LPCplex caller...", ioIn) {
   addArgument(BoolArgument(lpcplexParameter_.integerConstraint_, "ic", "integerconstraint", "use integer constraints"));
   addArgument(IntArgument<>(lpcplexParameter_.numberOfThreads_, "", "threads", "number of threads", lpcplexParameter_.numberOfThreads_));
   addArgument(BoolArgument(lpcplexParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(lpcplexParameter_.cutUp_, "", "cutup", "cut up", lpcplexParameter_.cutUp_));
   double timeout =604800.0;
   addArgument(DoubleArgument<>(lpcplexParameter_.timeLimit_,"","maxTime","maximal runtime in seconds",timeout)); //default 1 week
}

template <class IO, class GM, class ACC>
inline LPCplexCaller<IO, GM, ACC>::~LPCplexCaller() {

}

template <class IO, class GM, class ACC>
inline void LPCplexCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LPCplex caller" << std::endl;

   this-> template infer<LPCPLEX, TimingVisitorType, typename LPCPLEX::Parameter>(model, output, verbose, lpcplexParameter_);
}

template <class IO, class GM, class ACC>
const std::string LPCplexCaller<IO, GM, ACC>::name_ = "LPCPLEX";

} // namespace interface

} // namespace opengm

#endif /* LPCPLEX_CALLER_HXX_ */
