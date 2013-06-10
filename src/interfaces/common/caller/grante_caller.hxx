#ifndef GRANTE_CALLER_HXX_
#define GRANTE_CALLER_HXX_

#include <opengm/opengm.hxx>

#include <opengm/inference/external/grante.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class GranteCaller : public InferenceCallerBase<IO, GM, ACC, GranteCaller<IO, GM, ACC> > {
protected:
   typedef typename opengm::external::GRANTE<GM> GRANTE;
   typedef InferenceCallerBase<IO, GM, ACC, GranteCaller<IO, GM, ACC> > BaseClass;
   typedef typename GRANTE::VerboseVisitorType VerboseVisitorType;
   typedef typename GRANTE::EmptyVisitorType EmptyVisitorType;
   typedef typename GRANTE::TimingVisitorType TimingVisitorType;

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;

   typename GRANTE::Parameter granteParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
   std::string selectedInferenceType_;
   std::string selectedBPScheduleType_;
public:
   const static std::string name_;
   GranteCaller(IO& ioIn);
   virtual ~GranteCaller();
};

template <class IO, class GM, class ACC>
inline GranteCaller<IO, GM, ACC>::GranteCaller(IO& ioIn)
   : BaseClass("GRANTE", "detailed description of GRANTE Parser...", ioIn){

   std::vector<std::string> possibleInferenceTypes;
   possibleInferenceTypes.push_back("BRUTEFORCE");
   possibleInferenceTypes.push_back("BP");
   possibleInferenceTypes.push_back("DIFFUSION");
   possibleInferenceTypes.push_back("SA");
   addArgument(StringArgument<>(selectedInferenceType_, "", "inference", "Select desired GRANTE inference algorithm.", possibleInferenceTypes.front(), possibleInferenceTypes));

   std::vector<std::string> possibleBPScheduleTypes;
   possibleBPScheduleTypes.push_back("SEQUENTIAL");
   possibleBPScheduleTypes.push_back("PARALLELSYNC");
   addArgument(StringArgument<>(selectedBPScheduleType_, "", "BPSchedule", "Select MessageSchedule type for Belief Propagation method.", possibleBPScheduleTypes.front(), possibleBPScheduleTypes));

   addArgument(Size_TArgument<>(granteParameter_.numberOfIterations_, "", "maxIt", "Maximum number of iterations for Belief Propagation method.", granteParameter_.numberOfIterations_));
   addArgument(DoubleArgument<>(granteParameter_.tolerance_, "", "eps", "Used to define the threshold for stopping condition for Belief Propagation method.", granteParameter_.tolerance_));
   addArgument(BoolArgument(granteParameter_.verbose_, "", "statistics", "Print iteration statistics for Belief Propagation method"));

   addArgument(ArgumentBase<unsigned int>(granteParameter_.SASteps_, "", "SASteps", "Number of simulated annealing distributions.", granteParameter_.SASteps_));
   addArgument(DoubleArgument<>(granteParameter_.SAT0_, "", "SAT0", "Initial Boltzmann temperature for simulated annealing.", granteParameter_.SAT0_));
   addArgument(DoubleArgument<>(granteParameter_.SATfinal_, "", "SATfinal", "Final Boltzmann temperature for simulated annealing.", granteParameter_.SATfinal_));
}

template <class IO, class GM, class ACC>
inline GranteCaller<IO, GM, ACC>::~GranteCaller() {

}

template <class IO, class GM, class ACC>
inline void GranteCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running GRANTE caller" << std::endl;

   if(selectedInferenceType_ == "BRUTEFORCE") {
         granteParameter_.inferenceType_= GRANTE::Parameter::BRUTEFORCE;
   } else if(selectedInferenceType_ == "BP") {
      granteParameter_.inferenceType_= GRANTE::Parameter::BP;
   } else if(selectedInferenceType_ == "DIFFUSION") {
         granteParameter_.inferenceType_= GRANTE::Parameter::DIFFUSION;
   } else if(selectedInferenceType_ == "SA") {
         granteParameter_.inferenceType_= GRANTE::Parameter::SA;
   } else {
     throw RuntimeError("Unknown inference type for GRANTE");
   }

   if(selectedBPScheduleType_ == "SEQUENTIAL") {
      granteParameter_.BPSchedule_= Grante::BeliefPropagation::Sequential;
   } else if(selectedBPScheduleType_ == "PARALLELSYNC") {
      granteParameter_.BPSchedule_= Grante::BeliefPropagation::ParallelSync;
   } else {
     throw RuntimeError("Unknown BPSchedule type for GRANTE");
   }

   this-> template infer<GRANTE, TimingVisitorType, typename GRANTE::Parameter>(model, output, verbose, granteParameter_);
}

template <class IO, class GM, class ACC>
const std::string GranteCaller<IO, GM, ACC>::name_ = "GRANTE";

} // namespace interface

} // namespace opengm


#endif /* GRANTE_CALLER_HXX_ */
