#ifndef LPGUROBI_CALLER_HXX_
#define LPGUROBI_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/auxiliary/lp_solver/lp_solver_gurobi.hxx>
#include <opengm/inference/lp_inference.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LPGurobiCaller : public InferenceCallerBase<IO, GM, ACC, LPGurobiCaller<IO, GM, ACC> > {
public:
   typedef opengm::LpSolverGurobi LpSolver;
   typedef LPInference<GM, ACC,LpSolver> LPGUROBI;
   typedef InferenceCallerBase<IO, GM, ACC, LPGurobiCaller<IO, GM, ACC> > BaseClass;
   typedef typename LPGUROBI::VerboseVisitorType VerboseVisitorType;
   typedef typename LPGUROBI::EmptyVisitorType EmptyVisitorType;
   typedef typename LPGUROBI::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LPGurobiCaller(IO& ioIn);
   virtual ~LPGurobiCaller();
protected:

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename LPGUROBI::Parameter parameter_;

};

template <class IO, class GM, class ACC>
inline LPGurobiCaller<IO, GM, ACC>::LPGurobiCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LPGurobi caller...", ioIn) {
   addArgument(BoolArgument(parameter_.integerConstraint_, "ic", "integerconstraint", "use integer constraints"));
   //addArgument(IntArgument<>(lpcplexParameter_.numberOfThreads_, "", "threads", "number of threads", lpcplexParameter_.numberOfThreads_));
   //addArgument(BoolArgument(lpcplexParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   //addArgument(DoubleArgument<>(lpcplexParameter_.cutUp_, "", "cutup", "cut up", lpcplexParameter_.cutUp_));
   //double timeout =604800.0;
   //addArgument(DoubleArgument<>(lpcplexParameter_.timeLimit_,"","timeout","maximal runtime in seconds",timeout)); //default 1 week
}

template <class IO, class GM, class ACC>
inline LPGurobiCaller<IO, GM, ACC>::~LPGurobiCaller() {

}

template <class IO, class GM, class ACC>
inline void LPGurobiCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LPGurobi caller" << std::endl;

   this-> template infer<LPGUROBI, TimingVisitorType, typename LPGUROBI::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LPGurobiCaller<IO, GM, ACC>::name_ = "LPGUROBI";

} // namespace interface

} // namespace opengm

#endif /* LPGUROBI_CALLER_HXX_ */
