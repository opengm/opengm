#ifndef ALPHAEXPANSION_CALLER_HXX_
#define ALPHAEXPANSION_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/alphaexpansion.hxx>
#include <opengm/inference/graphcut.hxx>

#include "graphcut_caller.hxx"
#include "../argument/argument.hxx"

#ifdef WITH_BOOST
#  include <opengm/inference/auxiliary/minstcutboost.hxx>
#endif

#ifdef WITH_MAXFLOW
#  include <opengm/inference/auxiliary/minstcutkolmogorov.hxx>
#endif

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class AlphaExpansionCaller : public GraphCutCaller<IO, GM, ACC, AlphaExpansionCaller<IO, GM, ACC> > {
public:
   typedef GraphCutCaller<IO, GM, ACC, AlphaExpansionCaller<IO, GM, ACC> > BaseClass;
   const static std::string name_;
   AlphaExpansionCaller(IO& ioIn);
   virtual ~AlphaExpansionCaller();

   friend class GraphCutCaller<IO, GM, ACC, AlphaExpansionCaller<IO, GM, ACC> >;

protected:

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::scale_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   template <class MINSTCUT>
   void runImplHelper(GM& model, OutputBase& output, const bool verbose);
   size_t maxNumberOfSteps_;
   size_t randSeedOrder_;
   size_t randSeedLabel_;
   std::vector<typename GM::LabelType> labelOrder_;
   std::vector<typename GM::LabelType> label_;
   std::string desiredLabelInitialType_;
   std::string desiredOrderType_;
};

template <class IO, class GM, class ACC>
inline AlphaExpansionCaller<IO, GM, ACC>::AlphaExpansionCaller(IO& ioIn)
   : BaseClass(ioIn, name_, "detailed description of AlphaExpansion caller...") {
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
inline AlphaExpansionCaller<IO, GM, ACC>::~AlphaExpansionCaller() {

}

template <class IO, class GM, class ACC>
template <class MINSTCUT>
void AlphaExpansionCaller<IO, GM, ACC>::runImplHelper(GM& model, OutputBase& output, const bool verbose) {
   typedef GraphCut<GM, ACC, MINSTCUT> GraphCut;
   typename GraphCut::Parameter graphcutparameter;
   graphcutparameter.scale_ = scale_;

   typedef AlphaExpansion<GM, GraphCut> AlphaExpansion;
   typename AlphaExpansion::Parameter alphaexpansionparameter;
   alphaexpansionparameter.parameter_ = graphcutparameter;
   alphaexpansionparameter.maxNumberOfSteps_ = maxNumberOfSteps_;
   alphaexpansionparameter.randSeedOrder_ = randSeedOrder_;
   alphaexpansionparameter.randSeedLabel_ = randSeedLabel_;
   alphaexpansionparameter.labelOrder_ = labelOrder_;
   alphaexpansionparameter.label_ = label_;

   //LabelInitialType
   if(desiredLabelInitialType_ == "DEFAULT") {
      alphaexpansionparameter.labelInitialType_ = AlphaExpansion::Parameter::DEFAULT_LABEL;
   } else if(desiredLabelInitialType_ == "RANDOM") {
      alphaexpansionparameter.labelInitialType_ = AlphaExpansion::Parameter::RANDOM_LABEL;
   } else if(desiredLabelInitialType_ == "LOCALOPT") {
      alphaexpansionparameter.labelInitialType_ = AlphaExpansion::Parameter::LOCALOPT_LABEL;
   } else if(desiredLabelInitialType_ == "EXPLICIT") {
      alphaexpansionparameter.labelInitialType_ = AlphaExpansion::Parameter::EXPLICIT_LABEL;
   } else {
      throw RuntimeError("Unknown initial label type!");
   }

   //orderType
   if(desiredOrderType_ == "DEFAULT") {
      alphaexpansionparameter.orderType_ = AlphaExpansion::Parameter::DEFAULT_ORDER;
   } else if(desiredOrderType_ == "RANDOM") {
      alphaexpansionparameter.orderType_ = AlphaExpansion::Parameter::RANDOM_ORDER;
   } else if(desiredOrderType_ == "EXPLICIT") {
      alphaexpansionparameter.orderType_ = AlphaExpansion::Parameter::EXPLICIT_ORDER;
   } else {
      throw RuntimeError("Unknown order type!");
   }

   typedef typename AlphaExpansion::VerboseVisitorType CurrentVerboseVisitorType;
   typedef typename AlphaExpansion::EmptyVisitorType CurrentEmptyVisitorType;
   typedef typename AlphaExpansion::TimingVisitorType CurrentTimingVisitorType;

   this-> template infer<AlphaExpansion, CurrentTimingVisitorType, typename AlphaExpansion::Parameter>(model, output, verbose, alphaexpansionparameter);
}

template <class IO, class GM, class ACC>
const std::string AlphaExpansionCaller<IO, GM, ACC>::name_ = "ALPHAEXPANSION";

} // namespace interface

} // namespace opengm

#endif /* ALPHAEXPANSION_CALLER_HXX_ */
