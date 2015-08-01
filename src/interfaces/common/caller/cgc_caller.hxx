
#ifndef CGC_CALLER2_HXX_
#define CGC_CALLER2_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/cgc.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class CgcCaller : public InferenceCallerBase<IO, GM, ACC, CgcCaller<IO, GM, ACC> > {
public:
   typedef CGC<GM, ACC> Solver;
   typedef InferenceCallerBase<IO, GM, ACC, CgcCaller<IO, GM, ACC> > BaseClass;
   typedef typename Solver::VerboseVisitorType  VerboseVisitorType;
   typedef typename Solver::EmptyVisitorType    EmptyVisitorType;
   typedef typename Solver::TimingVisitorType   TimingVisitorType;

   const static std::string name_;
   CgcCaller(IO& ioIn);
   virtual ~CgcCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename Solver::Parameter param_;
   std::string selectedSolverType_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);



};

template <class IO, class GM, class ACC>
inline CgcCaller<IO, GM, ACC>::CgcCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of CgcCaller caller...", ioIn) {


   addArgument(DoubleArgument<>(param_.threshold_,
      "th", "threshold", "starting point merge threshold", param_.threshold_));

   addArgument(BoolArgument(param_.planar_, 
      "", "planar", "is model planar"));


   addArgument(BoolArgument(param_.useBookkeeping_, 
      "ub", "useBookkeeping", "use useBookkeeping ?"));


   addArgument(Size_TArgument<>(param_.maxIterations_, "mi","maxIterations",
                                "number of maximal iterations", param_.maxIterations_));


}

template <class IO, class GM, class ACC>
inline CgcCaller<IO, GM, ACC>::~CgcCaller() {

}

template <class IO, class GM, class ACC>
inline void CgcCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   //TK // std::cout << "running hmc caller" << std::endl;
   this-> template infer<Solver, TimingVisitorType, typename Solver::Parameter>(model, output, verbose, param_);
}

template <class IO, class GM, class ACC>
const std::string CgcCaller<IO, GM, ACC>::name_ = "CGC";

} // namespace interface

} // namespace opengm

#endif /* CGC_CALLER2_HXX_ */
