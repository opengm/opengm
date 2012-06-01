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
      class TRWSCaller : public InferenceCallerBase<IO, GM, ACC> {
      protected:
         using InferenceCallerBase<IO, GM, ACC>::addArgument;
         using InferenceCallerBase<IO, GM, ACC>::io_;
         using InferenceCallerBase<IO, GM, ACC>::infer;
         typedef typename opengm::external::TRWS<GM> TRWS;
         typename TRWS::Parameter trwsParameter_;
         typedef typename TRWS::VerboseVisitorType VerboseVisitorType;
         typedef typename TRWS::EmptyVisitorType EmptyVisitorType;
         typedef typename TRWS::TimingVisitorType TimingVisitorType; 
         virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
         std::string selectedEnergyType_;
      public:
         const static std::string name_;
         TRWSCaller(IO& ioIn);
      };

      template <class IO, class GM, class ACC>
      inline TRWSCaller<IO, GM, ACC>::TRWSCaller(IO& ioIn)
         : InferenceCallerBase<IO, GM, ACC>("TRWS", "detailed description of TRWS Parser...", ioIn)
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
         addArgument(BoolArgument(trwsParameter_.doBPS_, "", "doBPS", "Do BPS instead of TRWS"));
      }

      template <class IO, class GM, class ACC>
      inline void TRWSCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
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

         this-> template infer<TRWS, TimingVisitorType, typename TRWS::Parameter>(model, outputfile, verbose, trwsParameter_);
/*

         typename opengm::external::TRWS<GM>::Parameter parameter; 
         parameter.numberOfIterations_ = numberOfIterations_;
         opengm::external::TRWS<GM> trws(model, parameter);
         opengm::external::TRWSVisitor<opengm::external::TRWS<GM> > visitor;  
         std::vector<size_t> states;
         std::cout << "Inferring!" << std::endl;
         if(!(trws.infer(visitor) == opengm::NORMAL)) {
            std::string error("TRWS did not solve the problem.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }
         std::cout << "writing states in vector!" << std::endl;
         if(!(trws.arg(states) == opengm::NORMAL)) {
            std::string error("TRWS could not return optimal argument.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }  
         //storeVector(outputfile, states);
         storeVectorHDF5(outputfile,"states", states);
         storeVectorHDF5(outputfile,"values", visitor.values());
         storeVectorHDF5(outputfile,"bounds", visitor.bounds());
         storeVectorHDF5(outputfile,"times",  visitor.times());
         storeVectorHDF5(outputfile,"primalTimes",  visitor.primalTimes());
         storeVectorHDF5(outputfile,"dualTimes",  visitor.dualTimes());
        */
      }

      template <class IO, class GM, class ACC>
      const std::string TRWSCaller<IO, GM, ACC>::name_ = "TRWS";

   } // namespace interface

} // namespace opengm

#endif /* TRWS_CALLER_HXX_ */
