#ifndef MRFLIB_CALLER_HXX_
#define MRFLIB_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/mrflib.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MRFLIBCaller : public InferenceCallerBase<IO, GM, ACC, MRFLIBCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::external::MRFLIB<GM> MRFLIB;
   typedef InferenceCallerBase<IO, GM, ACC, MRFLIBCaller<IO, GM, ACC> > BaseClass;
   typedef typename MRFLIB::VerboseVisitorType VerboseVisitorType;
   typedef typename MRFLIB::EmptyVisitorType EmptyVisitorType;
   typedef typename MRFLIB::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   MRFLIBCaller(IO& ioIn);
   virtual ~MRFLIBCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typename MRFLIB::Parameter mrfParameter_;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   std::string selectedInferenceType_;
   std::string selectedEnergyType_;
};

template <class IO, class GM, class ACC>
inline MRFLIBCaller<IO, GM, ACC>::MRFLIBCaller(IO& ioIn)
   : BaseClass("MRF", "detailed description of MRF Parser...", ioIn) {
   std::vector<std::string> possibleInferenceTypes;
   possibleInferenceTypes.push_back("ICM");
   possibleInferenceTypes.push_back("EXPANSION");
   possibleInferenceTypes.push_back("SWAP");
   possibleInferenceTypes.push_back("MAXPRODBP");
   possibleInferenceTypes.push_back("TRWS");
   possibleInferenceTypes.push_back("BPS");
   addArgument(StringArgument<>(selectedInferenceType_, "", "inference", "Select desired MRF inference algorithm.", possibleInferenceTypes.front(), possibleInferenceTypes));

   std::vector<std::string> possibleEnergyTypes;
   possibleEnergyTypes.push_back("VIEW");
   possibleEnergyTypes.push_back("TABLES");
   possibleEnergyTypes.push_back("TL1");
   possibleEnergyTypes.push_back("TL2");
   possibleEnergyTypes.push_back("WEIGHTEDTABLE");
   addArgument(StringArgument<>(selectedEnergyType_, "", "energy", "Select desired MRF energy type.", possibleEnergyTypes.front(), possibleEnergyTypes));

   addArgument(Size_TArgument<>(mrfParameter_.numberOfIterations_, "", "maxIt", "Maximum number of iterations.", mrfParameter_.numberOfIterations_));
}

template <class IO, class GM, class ACC>
inline MRFLIBCaller<IO, GM, ACC>::~MRFLIBCaller() {

}

template <class IO, class GM, class ACC>
inline void MRFLIBCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running MRF caller" << std::endl;

   if(selectedInferenceType_ == "ICM") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::ICM;
   } else if(selectedInferenceType_ == "EXPANSION") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::EXPANSION;
   } else if(selectedInferenceType_ == "SWAP") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::SWAP;
   } else if(selectedInferenceType_ == "MAXPRODBP") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::MAXPRODBP;
   } else if(selectedInferenceType_ == "TRWS") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::TRWS;
   } else if(selectedInferenceType_ == "BPS") {
      mrfParameter_.inferenceType_= MRFLIB::Parameter::BPS;
   } else {
     throw RuntimeError("Unknown inference type for MRF");
   }

   if(selectedEnergyType_ == "VIEW") {
      mrfParameter_.energyType_= MRFLIB::Parameter::VIEW;
   } else if(selectedEnergyType_ == "TABLES") {
      mrfParameter_.energyType_= MRFLIB::Parameter::TABLES;
   } else if(selectedEnergyType_ == "TL1") {
      mrfParameter_.energyType_= MRFLIB::Parameter::TL1;
   } else if(selectedEnergyType_ == "TL2") {
      mrfParameter_.energyType_= MRFLIB::Parameter::TL2;
   } else if(selectedEnergyType_ == "WEIGHTEDTABLE") {
      mrfParameter_.energyType_= MRFLIB::Parameter::WEIGHTEDTABLE;
   } else {
     throw RuntimeError("Unknown energy type for MRF");
   }
   this-> template infer<MRFLIB, TimingVisitorType, typename MRFLIB::Parameter>(model, output, verbose, mrfParameter_);
}

template <class IO, class GM, class ACC>
const std::string MRFLIBCaller<IO, GM, ACC>::name_ = "MRF";

} // namespace interface

} // namespace opengm
#endif /* MRFLIB_CALLER_HXX_ */
