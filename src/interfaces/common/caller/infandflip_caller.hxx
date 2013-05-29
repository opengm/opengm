#ifndef INFANDFLIP_CALLER_HXX_
#define INFANDFLIP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lazyflipper.hxx>
#include <opengm/inference/infandflip.hxx>
#include <opengm/inference/messagepassing/messagepassing.hxx>
#ifdef WITH_TRWS
#include <opengm/inference/external/trws.hxx>
#endif
#ifdef WITH_FASTPD
#include <opengm/inference/external/fastPD.hxx>
#endif
 

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class InfAndFlipCaller : public InferenceCallerBase<IO, GM, ACC, InfAndFlipCaller<IO, GM, ACC> > {
protected:
   typedef InferenceCallerBase<IO, GM, ACC, InfAndFlipCaller<IO, GM, ACC> > BaseClass;
   typedef typename BaseClass::OutputBase OutputBase;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   size_t maxSubgraphSize_;
   size_t numberOfIterations_;
   std::string selectedInfType_;
   std::string selectedEnergyType_;
public:
   const static std::string name_;
   InfAndFlipCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline InfAndFlipCaller<IO, GM, ACC>::InfAndFlipCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of InfAndFlipper caller...", ioIn) {
   std::vector<std::string> possibleInfTypes;
   possibleInfTypes.push_back("TRWS");
   possibleInfTypes.push_back("FASTPD");
   possibleInfTypes.push_back("LBP");
   std::vector<std::string> possibleEnergyTypes;
   possibleEnergyTypes.push_back("VIEW");
   possibleEnergyTypes.push_back("TABLES");
   possibleEnergyTypes.push_back("TL1");
   possibleEnergyTypes.push_back("TL2");

   addArgument(Size_TArgument<>(maxSubgraphSize_, "", "maxsize", "maximum sub-graph size", (size_t)2));
   addArgument(Size_TArgument<>(numberOfIterations_, "", "maxIt", "maximum number of itterations", (size_t)1000));
   addArgument(StringArgument<>(selectedInfType_, "", "inf", "Select desired inference type.", possibleInfTypes.front(), possibleInfTypes)); 
   addArgument(StringArgument<>(selectedEnergyType_, "", "energy", "Select desired energy type.", possibleEnergyTypes.front(), possibleEnergyTypes));
}

template <class IO, class GM, class ACC>
inline void InfAndFlipCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running InfAndFlipper caller" << std::endl;
 
   if(selectedInfType_ == "TRWS") {
#ifdef WITH_TRWS
      typedef external::TRWS<GM> INFType;
      typedef InfAndFlip<GM, ACC,INFType> InfAndFlipType;
      typedef typename InfAndFlipType::VerboseVisitorType VerboseVisitorType;
      typedef typename InfAndFlipType::EmptyVisitorType EmptyVisitorType;
      typedef typename InfAndFlipType::TimingVisitorType TimingVisitorType;
      typename InfAndFlipType::Parameter infandflipParameter_;
      infandflipParameter_.maxSubgraphSize_=maxSubgraphSize_;
      infandflipParameter_.subPara_.numberOfIterations_=numberOfIterations_;
      if(selectedEnergyType_ == "VIEW") {
         infandflipParameter_.subPara_.energyType_= INFType::Parameter::VIEW;
      } else if(selectedEnergyType_ == "TABLES") {
         infandflipParameter_.subPara_.energyType_= INFType::Parameter::TABLES;
      } else if(selectedEnergyType_ == "TL1") {
         infandflipParameter_.subPara_.energyType_= INFType::Parameter::TL1;
      } else if(selectedEnergyType_ == "TL2") {
         infandflipParameter_.subPara_.energyType_= INFType::Parameter::TL2;
      } else {
         throw RuntimeError("Unknown energy type for TRWS");
      }
      this-> template infer<InfAndFlipType, TimingVisitorType, typename InfAndFlipType::Parameter>(model, output, verbose, infandflipParameter_);
#else
      throw RuntimeError("TRWS is not enabled!");
#endif
   } else if(selectedInfType_ == "FASTPD") {
#ifdef WITH_FASTPD
      typedef typename external::FastPD<GM> INFType;
      typedef InfAndFlip<GM, ACC,INFType> InfAndFlipType;
      typedef typename InfAndFlipType::VerboseVisitorType VerboseVisitorType;
      typedef typename InfAndFlipType::EmptyVisitorType EmptyVisitorType;
      typedef typename InfAndFlipType::TimingVisitorType TimingVisitorType;
      typename InfAndFlipType::Parameter infandflipParameter_;
      infandflipParameter_.maxSubgraphSize_=maxSubgraphSize_;
      infandflipParameter_.subPara_.numberOfIterations_=numberOfIterations_;
      this-> template infer<InfAndFlipType, TimingVisitorType, typename InfAndFlipType::Parameter>(model, output, verbose, infandflipParameter_);
#else
      throw RuntimeError("FASTPD is not enabled!");
#endif
   } else if(selectedInfType_ == "LBP") {
      typedef BeliefPropagationUpdateRules<GM, ACC> UpdateRulesType;
      typedef MessagePassing<GM, ACC, UpdateRulesType> INFType;
      typedef InfAndFlip<GM, ACC,INFType> InfAndFlipType;
      typedef typename InfAndFlipType::VerboseVisitorType VerboseVisitorType;
      typedef typename InfAndFlipType::EmptyVisitorType EmptyVisitorType;
      typedef typename InfAndFlipType::TimingVisitorType TimingVisitorType;
      typename InfAndFlipType::Parameter infandflipParameter_;
      infandflipParameter_.maxSubgraphSize_=maxSubgraphSize_;
      infandflipParameter_.subPara_.maximumNumberOfSteps_=numberOfIterations_; 
      infandflipParameter_.subPara_.damping_ = 0.5;
      infandflipParameter_.subPara_.bound_= 0.00001;
      this-> template infer<InfAndFlipType, TimingVisitorType, typename InfAndFlipType::Parameter>(model, output, verbose, infandflipParameter_);
      
   } else {
      throw RuntimeError("Unknown inference type");
   }
   
}

template <class IO, class GM, class ACC>
const std::string InfAndFlipCaller<IO, GM, ACC>::name_ = "INFANDFLIP";

} // namespace interface

} // namespace opengm

#endif /* INFANDFLIP_CALLER_HXX_ */
