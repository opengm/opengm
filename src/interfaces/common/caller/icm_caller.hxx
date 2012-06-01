#ifndef ICM_CALLER_HXX_
#define ICM_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/icm.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class ICMCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
/*   using InferenceCallerBase<IO, GM, ACC>::protocolate_;
   using InferenceCallerBase<IO, GM, ACC>::protocolate;
   using InferenceCallerBase<IO, GM, ACC>::store;*/
   typedef typename opengm::ICM<GM, ACC> ICM;
   typename ICM::Parameter icmParameter_;
   typedef typename ICM::VerboseVisitorType VerboseVisitorType;
   typedef typename ICM::EmptyVisitorType EmptyVisitorType;
   typedef typename ICM::TimingVisitorType TimingVisitorType;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
public:
   const static std::string name_;
   ICMCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline ICMCaller<IO, GM, ACC>::ICMCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>("ICM", "detailed description of ICM Parser...", ioIn) {
   addArgument(VectorArgument<std::vector<size_t> >(icmParameter_.startPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline void ICMCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running ICM caller" << std::endl;

   this-> template infer<ICM, TimingVisitorType, typename ICM::Parameter>(model, outputfile, verbose, icmParameter_);
/*
   opengm::ICM<GM, ACC> icm(model, icmParameter_);

   if(protocolate_->isSet()) {
      if(protocolate_->getValue() != 0) {
         std::cout << "N: " << protocolate_->getValue() << std::endl;
         TimingVisitorType visitor(protocolate_->getValue(), 0, verbose);
         if(!(icm.infer(visitor) == NORMAL)) {
            std::string error("ICM did not solve the problem.");
            io_.errorStream() << error << std::endl;
            throw RuntimeError(error);
         }
         if(outputfile.isSet()) {
            protocolate(visitor, outputfile.getValue());
         }
      } else {

      }
   } else {
      EmptyVisitorType visitor;
      if(!(icm.infer(visitor) == NORMAL)) {
         std::string error("ICM did not solve the problem.");
         io_.errorStream() << error << std::endl;
         throw RuntimeError(error);
      }
   }
   if(outputfile.isSet()) {

      std::vector<size_t> states;
      if(!(icm.arg(states) == NORMAL)) {
         std::string error("ICM could not return optimal argument.");
         io_.errorStream() << error << std::endl;
         throw RuntimeError(error);
      }

      std::cout << "storing optimal states in file: " << outputfile.getValue() << std::endl;
      store(states, outputfile.getValue(), "states");
   }*/
}

template <class IO, class GM, class ACC>
const std::string ICMCaller<IO, GM, ACC>::name_ = "ICM";

} // namespace interface

} // namespace opengm

#endif /* ICM_CALLER_HXX_ */
