#ifndef MESSAGEPASSING_BP_CALLER_HXX_
#define MESSAGEPASSING_BP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/messagepassing/messagepassing_bp.hxx>

#include "messagepassing_caller.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MessagepassingBPCaller : public MessagepassingCaller<IO, GM, ACC, BeliefPropagationUpdateRules<GM, ACC> > {
protected:
   typedef BeliefPropagationUpdateRules<GM, ACC> UpdateRulesType;
   typedef typename MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::MP BP;
   using MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::parameter_;
   using MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   typedef typename BP::VerboseVisitorType VerboseVisitorType;
   typedef typename BP::EmptyVisitorType EmptyVisitorType;
   typedef typename BP::TimingVisitorType TimingVisitorType;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
public:
   const static std::string name_;
   MessagepassingBPCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline MessagepassingBPCaller<IO, GM, ACC>::MessagepassingBPCaller(IO& ioIn)
   : MessagepassingCaller<IO, GM, ACC, UpdateRulesType>(name_, "detailed description of Beliefpropagation caller...", ioIn) {
}

template <class IO, class GM, class ACC>
inline void MessagepassingBPCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running Beliefpropagation caller" << std::endl;

   this-> template infer<BP, TimingVisitorType, typename BP::Parameter>(model, outputfile, verbose, parameter_);
   /*
   BP beliefpropagation(model, parameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(beliefpropagation.infer() == NORMAL)) {
      std::string error("Beliefpropagation did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(beliefpropagation.arg(states) == NORMAL)) {
      std::string error("Beliefpropagation could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string MessagepassingBPCaller<IO, GM, ACC>::name_ = "BELIEFPROPAGATION";

} // namespace interface

} // namespace opengm

#endif /* MESSAGEPASSING_BP_CALLER_HXX_ */
