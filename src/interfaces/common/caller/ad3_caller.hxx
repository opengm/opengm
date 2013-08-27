
#ifndef AD3_CALLER_HXX_
#define AD3_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/ad3.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class Ad3Caller : public InferenceCallerBase<IO, GM, ACC, Ad3Caller<IO, GM, ACC> > {
public:
   typedef external::AD3Inf<GM, ACC> Solver;
   typedef InferenceCallerBase<IO, GM, ACC, Ad3Caller<IO, GM, ACC> > BaseClass;
   typedef typename Solver::VerboseVisitorType  VerboseVisitorType;
   typedef typename Solver::EmptyVisitorType    EmptyVisitorType;
   typedef typename Solver::TimingVisitorType   TimingVisitorType;

   const static std::string name_;
   Ad3Caller(IO& ioIn);
   virtual ~Ad3Caller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename Solver::Parameter param_;
   std::string selectedSolverType_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);


   size_t steps_;
};

template <class IO, class GM, class ACC>
inline Ad3Caller<IO, GM, ACC>::Ad3Caller(IO& ioIn)
   : BaseClass(name_, "detailed description of Ad3Caller caller...", ioIn) {

   addArgument(DoubleArgument<>(param_.eta_, "", "eta", "eta.", double(param_.eta_)));
   addArgument(BoolArgument(param_.adaptEta_, "", "adaptEta", "adaptEta"));
   addArgument(Size_TArgument<>(steps_, "", "steps", "maximum steps", size_t(param_.steps_)));

   addArgument(DoubleArgument<>(param_.residualThreshold_, "", "residualThreshold", "residualThreshold", double(param_.residualThreshold_)));
   addArgument(IntArgument<>(param_.verbosity_, "", "verbosity", "verbosity", int(param_.verbosity_)));



   std::vector<std::string> possibleSolverType;
   possibleSolverType.push_back(std::string("AD3_LP"));
   possibleSolverType.push_back(std::string("AD3_ILP"));
   possibleSolverType.push_back(std::string("PSDD_LP"));

   addArgument(StringArgument<>(selectedSolverType_, "", "solverType", "selects the update rule", possibleSolverType.at(0), possibleSolverType));
}

template <class IO, class GM, class ACC>
inline Ad3Caller<IO, GM, ACC>::~Ad3Caller() {

}

template <class IO, class GM, class ACC>
inline void Ad3Caller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Ad3Caller caller" << std::endl;

   if(selectedSolverType_ == std::string("AD3_LP")) {
     param_.solverType_= Solver::AD3_LP;
   }
   else if(selectedSolverType_ == std::string("AD3_ILP")) {
     param_.solverType_= Solver::AD3_ILP;
   } 
   else if(selectedSolverType_ == std::string("PSDD_LP")) {
     param_.solverType_= Solver::PSDD_LP;
   }
   else {
     throw RuntimeError("Unknown solverType for ad3");
   }  
   param_.steps_=steps_;
   this-> template infer<Solver, TimingVisitorType, typename Solver::Parameter>(model, output, verbose, param_);
}

template <class IO, class GM, class ACC>
const std::string Ad3Caller<IO, GM, ACC>::name_ = "AD3";

} // namespace interface

} // namespace opengm

#endif /* AD3_CALLER_HXX_ */
