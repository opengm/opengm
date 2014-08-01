#ifndef LSATR_CALLER_HXX_
#define LSATR_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lsatr.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LSA_TRCaller : public InferenceCallerBase<IO, GM, ACC, LSA_TRCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::LSA_TR<GM, ACC> LSA_TR;
   typedef InferenceCallerBase<IO, GM, ACC, LSA_TRCaller<IO, GM, ACC> > BaseClass;
   typedef typename LSA_TR::VerboseVisitorType VerboseVisitorType;
   typedef typename LSA_TR::EmptyVisitorType EmptyVisitorType;
   typedef typename LSA_TR::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   LSA_TRCaller(IO& ioIn);
   virtual ~LSA_TRCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   std::string selectedDistanceType_;

   typedef typename BaseClass::OutputBase OutputBase;

   typename LSA_TR::Parameter parameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline LSA_TRCaller<IO, GM, ACC>::LSA_TRCaller(IO& ioIn)
   : BaseClass("LSA_TR", "detailed description of LSA_TR Parser...", ioIn) {
//   addArgument(VectorArgument<std::vector<typename LSA_TR::LabelType> >(parameter_.startPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
   addArgument(Size_TArgument<>(parameter_.randSeed_, "", "seed", "Seed for random generator.", (size_t)42));
   std::vector<std::string> possibleDistanceTypes;
   possibleDistanceTypes.push_back("HAMMING");
   possibleDistanceTypes.push_back("EUCLIDEAN");
   addArgument(StringArgument<>(selectedDistanceType_, "", "distance", "distance to current label used for trust region term.", possibleDistanceTypes.front(), possibleDistanceTypes));

}

template <class IO, class GM, class ACC>
inline LSA_TRCaller<IO, GM, ACC>::~LSA_TRCaller() {

}

template <class IO, class GM, class ACC>
inline void LSA_TRCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LSA_TR caller" << std::endl; 

   if(selectedDistanceType_ == "HAMMING") {
      parameter_.distance_= LSA_TR::Parameter::HAMMING;
   } else if(selectedDistanceType_ == "EUCLIDEAN") { 
      parameter_.distance_= LSA_TR::Parameter::EUCLIDEAN;
   }else{
      parameter_.distance_= LSA_TR::Parameter::HAMMING;
   }

   this-> template infer<LSA_TR, TimingVisitorType, typename LSA_TR::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LSA_TRCaller<IO, GM, ACC>::name_ = "LSA_TR";

} // namespace interface

} // namespace opengm

#endif /* LSATR_CALLER_HXX_ */
