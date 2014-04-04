#ifndef TRWS_CALLER_HXX_
#define TRWS_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/trws.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"
//#include "../helper/helper.hxx"

namespace opengm {

   namespace interface {

      template <class IO, class GM, class ACC>
      class TRWSCaller : public InferenceCallerBase<IO, GM, ACC, TRWSCaller<IO, GM, ACC> > {
      public:
         typedef typename opengm::external::TRWS<GM> TRWS;
         typedef InferenceCallerBase<IO, GM, ACC, TRWSCaller<IO, GM, ACC> > BaseClass;
         typedef typename TRWS::VerboseVisitorType VerboseVisitorType;
         typedef typename TRWS::EmptyVisitorType EmptyVisitorType;
         typedef typename TRWS::TimingVisitorType TimingVisitorType; 
         const static std::string name_;
         TRWSCaller(IO& ioIn);
         virtual ~TRWSCaller();
      protected:
         using BaseClass::addArgument;
         using BaseClass::io_;
         using BaseClass::infer;

         typedef typename BaseClass::OutputBase OutputBase;

         virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
         typename TRWS::Parameter trwsParameter_;
         std::string selectedEnergyType_;
      };

      template <class IO, class GM, class ACC>
      inline TRWSCaller<IO, GM, ACC>::TRWSCaller(IO& ioIn)
         : BaseClass("TRWS", "detailed description of TRWS Parser...", ioIn)
      {
         std::vector<std::string> possibleEnergyTypes;
         possibleEnergyTypes.push_back("VIEW");
         possibleEnergyTypes.push_back("TABLES");
         possibleEnergyTypes.push_back("TL1");
         possibleEnergyTypes.push_back("TL2");
         //possibleEnergyTypes.push_back("WEIGHTEDTABLE");
         addArgument(StringArgument<>(selectedEnergyType_, "", "energy", "Select desired energy type.", possibleEnergyTypes.front(), possibleEnergyTypes));

         addArgument(Size_TArgument<>(trwsParameter_.numberOfIterations_, "", "maxIt", "Maximum number of iterations.", trwsParameter_.numberOfIterations_));
         addArgument(BoolArgument(trwsParameter_.useRandomStart_, "", "randomStart", "Use random starting message."));
         addArgument(BoolArgument(trwsParameter_.useZeroStart_, "", "zeroStart", "Use zero starting message."));
         addArgument(BoolArgument(trwsParameter_.doBPS_, "", "doBPS", "Do BPS instead of TRWS"));  
         addArgument(DoubleArgument<>(trwsParameter_.minDualChange_, "", "minDualChange", "stop when change of dual is smaller",trwsParameter_.minDualChange_));
      }

      template <class IO, class GM, class ACC>
      inline TRWSCaller<IO, GM, ACC>::~TRWSCaller() {

      }

      template <class IO, class GM, class ACC>
      inline void TRWSCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
         std::cout << "running TRWS caller" << std::endl;

         if(selectedEnergyType_ == "VIEW") {
            trwsParameter_.energyType_= TRWS::Parameter::VIEW;
         } else if(selectedEnergyType_ == "TABLES") {
            trwsParameter_.energyType_= TRWS::Parameter::TABLES;
         } else if(selectedEnergyType_ == "TL1") {
            trwsParameter_.energyType_= TRWS::Parameter::TL1;
         } else if(selectedEnergyType_ == "TL2") {
            trwsParameter_.energyType_= TRWS::Parameter::TL2;
         /*} else if(selectedEnergyType_ == "WEIGHTEDTABLE") {
            trwsParameter_.energyType_= TRWS::Parameter::WEIGHTEDTABLE;*/
         } else {
           throw RuntimeError("Unknown energy type for TRWS");
         }

         this-> template infer<TRWS, TimingVisitorType, typename TRWS::Parameter>(model, output, verbose, trwsParameter_);
      }

      template <class IO, class GM, class ACC>
      const std::string TRWSCaller<IO, GM, ACC>::name_ = "TRWS";

   } // namespace interface

} // namespace opengm

#endif /* TRWS_CALLER_HXX_ */
