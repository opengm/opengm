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
public:
   typedef TrbpUpdateRules<GM, ACC> UpdateRulesType;
   typedef MessagepassingCaller<IO, GM, ACC, UpdateRulesType> BaseClass;

   const static std::string name_;
   MessagepassingTRBPCaller(IO& ioIn);
   virtual ~MessagepassingTRBPCaller();
protected:
   typedef typename BaseClass::MP TRBP;
   using BaseClass::parameter_;
   using BaseClass::addArgument;
   using BaseClass::infer;
   using BaseClass::io_;
   typedef typename BaseClass::OutputBase OutputBase;
   typedef typename TRBP::VerboseVisitorType VerboseVisitorType;
   typedef typename TRBP::EmptyVisitorType EmptyVisitorType;
   typedef typename TRBP::TimingVisitorType TimingVisitorType;

      virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline MessagepassingTRBPCaller<IO, GM, ACC>::MessagepassingTRBPCaller(IO& ioIn)
   : MessagepassingCaller<IO, GM, ACC, UpdateRulesType>(name_, "detailed description of TRBP caller...", ioIn) {
   addArgument(VectorArgument<typename TRBP::Parameter::SpecialParameterType>(parameter_.specialParameter_, "", "specialParameter", "Description for special parameter."));
}

template <class IO, class GM, class ACC>
inline MessagepassingTRBPCaller<IO, GM, ACC>::~MessagepassingTRBPCaller() {

}

template <class IO, class GM, class ACC>
inline void MessagepassingTRBPCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running TRBP caller" << std::endl;

   UpdateRulesType::initializeSpecialParameter(model, parameter_);

   this-> template infer<TRBP, TimingVisitorType, typename TRBP::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string MessagepassingTRBPCaller<IO, GM, ACC>::name_ = "TRBP";

} // namespace interface

} // namespace opengm

#endif /* MESSAGEPASSING_TRBP_CALLER_HXX_ */
