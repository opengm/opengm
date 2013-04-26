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
public:
   typedef BeliefPropagationUpdateRules<GM, ACC> UpdateRulesType;
   typedef MessagepassingCaller<IO, GM, ACC, UpdateRulesType> BaseClass;
   const static std::string name_;
   MessagepassingBPCaller(IO& ioIn);
   virtual ~MessagepassingBPCaller();
protected:
   typedef typename BaseClass::MP BP;
   using BaseClass::parameter_;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typedef typename BP::VerboseVisitorType VerboseVisitorType;
   typedef typename BP::EmptyVisitorType EmptyVisitorType;
   typedef typename BP::TimingVisitorType TimingVisitorType;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline MessagepassingBPCaller<IO, GM, ACC>::MessagepassingBPCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Beliefpropagation caller...", ioIn) {
}

template <class IO, class GM, class ACC>
inline MessagepassingBPCaller<IO, GM, ACC>::~MessagepassingBPCaller() {

}

template <class IO, class GM, class ACC>
inline void MessagepassingBPCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Beliefpropagation caller" << std::endl;

   this-> template infer<BP, TimingVisitorType, typename BP::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string MessagepassingBPCaller<IO, GM, ACC>::name_ = "BELIEFPROPAGATION";

} // namespace interface

} // namespace opengm

#endif /* MESSAGEPASSING_BP_CALLER_HXX_ */
