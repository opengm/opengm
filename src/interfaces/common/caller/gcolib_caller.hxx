#ifndef GCOLIB_CALLER_HXX_
#define GCOLIB_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/gco.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class GCOLIBCaller : public InferenceCallerBase<IO, GM, ACC, GCOLIBCaller<IO, GM, ACC> > {
public: 
   typedef typename opengm::external::GCOLIB<GM> GCOLIBType;
   typedef InferenceCallerBase<IO, GM, ACC, GCOLIBCaller<IO, GM, ACC> > BaseClass;
   typedef typename GCOLIBType::VerboseVisitorType VerboseVisitorType;
   typedef typename GCOLIBType::EmptyVisitorType EmptyVisitorType;
   typedef typename GCOLIBType::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   GCOLIBCaller(IO& ioIn);
   virtual ~GCOLIBCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 
   typedef typename BaseClass::OutputBase OutputBase;

   typename GCOLIBType::Parameter gcoParameter_;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
   std::string selectedInferenceType_;
   std::string selectedEnergyType_;
};

template <class IO, class GM, class ACC>
inline GCOLIBCaller<IO, GM, ACC>::GCOLIBCaller(IO& ioIn)
   : BaseClass("GCO", "detailed description of GCO Parser...", ioIn) {
   std::vector<std::string> possibleInferenceTypes;
   possibleInferenceTypes.push_back("EXPANSION");
   possibleInferenceTypes.push_back("SWAP");
   addArgument(StringArgument<>(selectedInferenceType_, "", "inference", "Select desired MRF inference algorithm.", possibleInferenceTypes.front(), possibleInferenceTypes));

   std::vector<std::string> possibleEnergyTypes;
   possibleEnergyTypes.push_back("VIEW");
   possibleEnergyTypes.push_back("TABLES");
   possibleEnergyTypes.push_back("WEIGHTEDTABLE");
   addArgument(StringArgument<>(selectedEnergyType_, "", "energy", "Select desired MRF energy type.", possibleEnergyTypes.front(), possibleEnergyTypes));

   addArgument(Size_TArgument<>(gcoParameter_.numberOfIterations_, "", "maxIt", "Maximum number of iterations.", gcoParameter_.numberOfIterations_));
   addArgument(BoolArgument(gcoParameter_.randomLabelOrder_, "", "randLabelOrder", "Use random label order."));
   addArgument(BoolArgument(gcoParameter_.useAdaptiveCycles_, "", "adaptiveCycles", "Use adaptive cycles for alpha-expansion."));
}

template <class IO, class GM, class ACC>
inline GCOLIBCaller<IO, GM, ACC>::~GCOLIBCaller() {

}

template <class IO, class GM, class ACC>
inline void GCOLIBCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running GCO caller" << std::endl;

   if(selectedInferenceType_ == "EXPANSION") {
      gcoParameter_.inferenceType_= GCOLIBType::Parameter::EXPANSION;
   } else if(selectedInferenceType_ == "SWAP") {
      gcoParameter_.inferenceType_= GCOLIBType::Parameter::SWAP;
   } else {
     throw RuntimeError("Unknown inference type for GCO");
   }

   if(selectedEnergyType_ == "VIEW") {
      gcoParameter_.energyType_= GCOLIBType::Parameter::VIEW;
   } else if(selectedEnergyType_ == "TABLES") {
      gcoParameter_.energyType_= GCOLIBType::Parameter::TABLES;
   } else if(selectedEnergyType_ == "WEIGHTEDTABLE") {
      gcoParameter_.energyType_= GCOLIBType::Parameter::WEIGHTEDTABLE;
   } else {
     throw RuntimeError("Unknown energy type for GCO");
   }
   this-> template infer<GCOLIBType, TimingVisitorType, typename GCOLIBType::Parameter>(model, output, verbose, gcoParameter_);
}

template <class IO, class GM, class ACC>
const std::string GCOLIBCaller<IO, GM, ACC>::name_ = "GCO";

} // namespace interface

} // namespace opengm


#endif /* GCOLIB_CALLER_HXX_ */
