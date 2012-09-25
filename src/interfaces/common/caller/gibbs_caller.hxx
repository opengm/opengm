#ifndef GIBBS_CALLER_HXX_
#define GIBBS_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/gibbs.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class GibbsCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef Gibbs<GM, ACC> GIBBS;
   typedef typename GIBBS::VerboseVisitorType VerboseVisitorType;
   typedef typename GIBBS::EmptyVisitorType EmptyVisitorType;
   typedef typename GIBBS::TimingVisitorType TimingVisitorType;
   typename GIBBS::Parameter gibbsParameter_; 
   std::string desiredVariableProposalType_;
public:
   const static std::string name_;
   GibbsCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline GibbsCaller<IO, GM, ACC>::GibbsCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of Gibbs caller...", ioIn) {
  
   addArgument(Size_TArgument<>(gibbsParameter_.maxNumberOfSamplingSteps_, "", "samplingSteps", "number of sampling steps", (size_t)1000));
   addArgument(Size_TArgument<>(gibbsParameter_.numberOfBurnInSteps_, "", "burninSteps", "number of burnin steps (should always be 0 for optimization)", (size_t)0));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(gibbsParameter_.startPoint_, "", "startPoint", "location of the file containing a vector which specifies the initial labeling", false));
   std::vector<std::string> permittedVariableProposalTypes;
   permittedVariableProposalTypes.push_back("CYCLIC");
   permittedVariableProposalTypes.push_back("RANDOM");
   addArgument(StringArgument<>(desiredVariableProposalType_, "", "variableOrder", "order in with variables are sampled", permittedVariableProposalTypes.at(0), permittedVariableProposalTypes));
}

template <class IO, class GM, class ACC>
inline void GibbsCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running Gibbs caller" << std::endl;

   //check start point
   if(gibbsParameter_.startPoint_.size() != model.numberOfVariables()){
      gibbsParameter_.startPoint_.resize(model.numberOfVariables(),0);
   }

   //orderType
   if(desiredVariableProposalType_ == "CYCLIC") {
      gibbsParameter_.variableProposal_ =  GIBBS::Parameter::CYCLIC; 
   } else if(desiredVariableProposalType_ == "RANDOM") {
      gibbsParameter_.variableProposal_ =  GIBBS::Parameter::RANDOM; 
   } else {
      throw RuntimeError("Unknown order type!");
   }

   this-> template infer<GIBBS, TimingVisitorType, typename GIBBS::Parameter>(model, outputfile, verbose, gibbsParameter_);
}

template <class IO, class GM, class ACC>
const std::string GibbsCaller<IO, GM, ACC>::name_ = "GIBBS";

} // namespace interface

} // namespace opengm

#endif /* GIBBS_CALLER_HXX_ */
