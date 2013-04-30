#ifndef MESSAGEPASSING_CALLER_HXX_
#define MESSAGEPASSING_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#include <opengm/inference/messagepassing/messagepassing_bp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC, class UPDATE_RULES>
class MessagepassingCaller : public InferenceCallerBase<IO, GM, ACC, MessagepassingCaller<IO, GM, ACC, UPDATE_RULES> > {
public:
   typedef MessagePassing<GM, ACC, UPDATE_RULES> MP;
   typedef InferenceCallerBase<IO, GM, ACC, MessagepassingCaller<IO, GM, ACC, UPDATE_RULES> > BaseClass;
   typedef typename MP::VerboseVisitorType VerboseVisitorType;
   typedef typename MP::EmptyVisitorType EmptyVisitorType;
   typedef typename MP::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   MessagepassingCaller(const std::string& InferenceParserNameIn, const std::string& inferenceParserDescriptionIn, IO& ioIn, const size_t maxNumArguments = 10);
   virtual ~MessagepassingCaller();
protected:

   typename MP::Parameter parameter_;
   using BaseClass::addArgument;
   using BaseClass::io_;
};

template <class IO, class GM, class ACC, class UPDATE_RULES>
inline MessagepassingCaller<IO, GM, ACC, UPDATE_RULES>::MessagepassingCaller(const std::string& MessagepassingCallerNameIn, const std::string& MessagepassingCallerDescriptionIn, IO& ioIn, const size_t maxNumArguments)
   : BaseClass(MessagepassingCallerNameIn, MessagepassingCallerDescriptionIn, ioIn, maxNumArguments + 3) {
   addArgument(Size_TArgument<>(parameter_.maximumNumberOfSteps_, "", "maxIt", "Maximum number of iterations.", static_cast<size_t>(100)));
   addArgument(ArgumentBase<typename GM::ValueType>(parameter_.bound_, "", "bound", "Add description for bound here!!!!.", typename GM::ValueType(0.0)));
   addArgument(ArgumentBase<typename GM::ValueType>(parameter_.damping_, "", "damping", "Add description for damping here!!!!.", typename GM::ValueType(0.0)));
}

template <class IO, class GM, class ACC, class UPDATE_RULES>
inline MessagepassingCaller<IO, GM, ACC, UPDATE_RULES>::~MessagepassingCaller() {

}

template <class IO, class GM, class ACC, class UPDATE_RULES>
const std::string MessagepassingCaller<IO, GM, ACC, UPDATE_RULES>::name_ = "MESSAGEPASSING";

} // namespace interface

} // namespace opengm

#endif /* MESSAGEPASSING_CALLER_HXX_ */
