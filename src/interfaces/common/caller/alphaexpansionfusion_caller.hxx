#ifndef ALPHAEXPANSIONFUSION_CALLER_HXX_
#define ALPHAEXPANSIONFUSION_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/alphaexpansionfusion.hxx>

#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class AlphaExpansionFusionCaller : public InferenceCallerBase<IO, GM, ACC, AlphaExpansionFusionCaller<IO, GM, ACC> > {

public: 
   typedef AlphaExpansionFusion<GM, ACC> AlphaExpansionFusionType;
   typedef InferenceCallerBase<IO, GM, ACC, AlphaExpansionFusionCaller<IO, GM, ACC> > BaseClass;
   typedef typename AlphaExpansionFusionType::VerboseVisitorType VerboseVisitorType;
   typedef typename AlphaExpansionFusionType::EmptyVisitorType EmptyVisitorType;
   typedef typename AlphaExpansionFusionType::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   AlphaExpansionFusionCaller(IO& ioIn);

protected:
    typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   size_t maxNumberOfSteps_;
   size_t randSeedOrder_;
   size_t randSeedLabel_;
   std::vector<typename GM::LabelType> labelOrder_;
   std::vector<typename GM::LabelType> label_;
   std::string desiredLabelInitialType_;
   std::string desiredOrderType_;
 
   void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline AlphaExpansionFusionCaller<IO, GM, ACC>::AlphaExpansionFusionCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Alpha-Expansion-Fusion caller...", ioIn) {
   addArgument(Size_TArgument<>(maxNumberOfSteps_, "", "maxIt", "Maximum number of iterations.", (size_t)1000));
   std::vector<std::string> permittedLabelInitialTypes;
   permittedLabelInitialTypes.push_back("DEFAULT");
   permittedLabelInitialTypes.push_back("RANDOM");
   permittedLabelInitialTypes.push_back("LOCALOPT");
   permittedLabelInitialTypes.push_back("EXPLICIT");
   addArgument(StringArgument<>(desiredLabelInitialType_, "", "labelInitialType", "select the desired initial label", permittedLabelInitialTypes.at(0), permittedLabelInitialTypes));
   std::vector<std::string> permittedOrderTypes;
   permittedOrderTypes.push_back("DEFAULT");
   permittedOrderTypes.push_back("RANDOM");
   permittedOrderTypes.push_back("EXPLICIT");
   addArgument(StringArgument<>(desiredOrderType_, "", "orderType", "select the desired order", permittedOrderTypes.at(0), permittedOrderTypes));
   addArgument(Size_TArgument<>(randSeedOrder_, "", "randSeedOrder", "Add description for randSeedOrder here!!!!.", (size_t)0));
   addArgument(Size_TArgument<>(randSeedLabel_, "", "randSeedLabel", "Add description for randSeedLabel here!!!!.", (size_t)0));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(labelOrder_, "", "labelorder", "location of the file containing a vector which specifies the desired label order", false));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(label_, "", "label", "location of the file containing a vector which specifies the desired label", false));

}

template <class IO, class GM, class ACC>
void AlphaExpansionFusionCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
 
   typename AlphaExpansionFusionType::Parameter parameter;
   parameter.maxNumberOfSteps_ = maxNumberOfSteps_;
   parameter.randSeedOrder_ = randSeedOrder_;
   parameter.randSeedLabel_ = randSeedLabel_;
   parameter.labelOrder_ = labelOrder_;
   parameter.label_ = label_;

   //LabelInitialType
   if(desiredLabelInitialType_ == "DEFAULT") {
      parameter.labelInitialType_ = AlphaExpansionFusionType::Parameter::DEFAULT_LABEL;
   } else if(desiredLabelInitialType_ == "RANDOM") {
      parameter.labelInitialType_ = AlphaExpansionFusionType::Parameter::RANDOM_LABEL;
   } else if(desiredLabelInitialType_ == "LOCALOPT") {
      parameter.labelInitialType_ = AlphaExpansionFusionType::Parameter::LOCALOPT_LABEL;
   } else if(desiredLabelInitialType_ == "EXPLICIT") {
      parameter.labelInitialType_ = AlphaExpansionFusionType::Parameter::EXPLICIT_LABEL;
   } else {
      throw RuntimeError("Unknown initial label type!");
   }

   //orderType
   if(desiredOrderType_ == "DEFAULT") {
      parameter.orderType_ = AlphaExpansionFusionType::Parameter::DEFAULT_ORDER;
   } else if(desiredOrderType_ == "RANDOM") {
      parameter.orderType_ = AlphaExpansionFusionType::Parameter::RANDOM_ORDER;
   } else if(desiredOrderType_ == "EXPLICIT") {
      parameter.orderType_ = AlphaExpansionFusionType::Parameter::EXPLICIT_ORDER;
   } else {
      throw RuntimeError("Unknown order type!");
   }


   this-> template infer<AlphaExpansionFusionType, TimingVisitorType, typename AlphaExpansionFusionType::Parameter>(model, output, verbose, parameter);

 
}

template <class IO, class GM, class ACC>
const std::string AlphaExpansionFusionCaller<IO, GM, ACC>::name_ = "ALPHAEXPANSIONFUSION";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSIONFUAION_CALLER_HXX_ */
