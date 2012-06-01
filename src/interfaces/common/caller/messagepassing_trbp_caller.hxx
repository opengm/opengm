#ifndef MESSAGEPASSING_TRBP_CALLER_HXX_
#define MESSAGEPASSING_TRBP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/messagepassing/messagepassing_trbp.hxx>
#include "messagepassing_caller.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MessagepassingTRBPCaller : public MessagepassingCaller<IO, GM, ACC, TrbpUpdateRules<GM, ACC> > {
protected:
   typedef TrbpUpdateRules<GM, ACC> UpdateRulesType;
   typedef typename MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::MP TRBP;
   using MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::parameter_;
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   using MessagepassingCaller<IO, GM, ACC, UpdateRulesType>::io_;
   typedef typename TRBP::VerboseVisitorType VerboseVisitorType;
   typedef typename TRBP::EmptyVisitorType EmptyVisitorType;
   typedef typename TRBP::TimingVisitorType TimingVisitorType;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
public:
   const static std::string name_;
   MessagepassingTRBPCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline MessagepassingTRBPCaller<IO, GM, ACC>::MessagepassingTRBPCaller(IO& ioIn)
   : MessagepassingCaller<IO, GM, ACC, UpdateRulesType>(name_, "detailed description of TRBP caller...", ioIn) {
   addArgument(VectorArgument<typename TRBP::Parameter::SpecialParameterType>(parameter_.specialParameter_, "", "specialParameter", "Description for special parameter."));
}

template <class IO, class GM, class ACC>
inline void MessagepassingTRBPCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running TRBP caller" << std::endl;

   UpdateRulesType::initializeSpecialParameter(model, parameter_);

   this-> template infer<TRBP, TimingVisitorType, typename TRBP::Parameter>(model, outputfile, verbose, parameter_);
/*   TRBP trbp(model, parameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(trbp.infer() == NORMAL)) {
      std::string error("TRBP did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(trbp.arg(states) == NORMAL)) {
      std::string error("TRBP could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string MessagepassingTRBPCaller<IO, GM, ACC>::name_ = "TRBP";

} // namespace interface

} // namespace opengm

#endif /* MESSAGEPASSING_TRBP_CALLER_HXX_ */
