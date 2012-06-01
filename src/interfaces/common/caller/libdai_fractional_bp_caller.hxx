#ifndef LIBDAI_FRACTIONAL_BP_CALLER
#define LIBDAI_FRACTIONAL_BP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/fractional_bp.hxx>
#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiFractionalBpCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef external::libdai::Bp<GM, ACC> LibDai_FractionalBp;
   typedef typename LibDai_FractionalBp::VerboseVisitorType VerboseVisitorType;
   typedef typename LibDai_FractionalBp::EmptyVisitorType EmptyVisitorType;
   typedef typename LibDai_FractionalBp::TimingVisitorType TimingVisitorType;
   typename LibDai_FractionalBp::Parameter parameter_;
   std::string selectedUpdateRule_;
public:
   const static std::string name_;
   LibDaiFractionalBpCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LibDaiFractionalBpCaller<IO, GM, ACC>::LibDaiFractionalBpCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LibDaiFractionalBpCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(parameter_.maxIterations_, "", "maxIt", "Maximum number of iterations.", size_t(parameter_.maxIterations_)));
   addArgument(DoubleArgument<>(parameter_.tolerance_, "", "bound", "convergence bound.", double(parameter_.tolerance_)));
   addArgument(DoubleArgument<>(parameter_.damping_, "", "damping", "message damping", double(0.0)));
   addArgument(Size_TArgument<>(parameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(parameter_.verbose_)));
   std::vector<std::string> possibleUpdateRule;
   possibleUpdateRule.push_back(std::string("PARALL"));
   possibleUpdateRule.push_back(std::string("SEQFIX"));
   possibleUpdateRule.push_back(std::string("SEQRND"));
   possibleUpdateRule.push_back(std::string("SEQMAX"));
   addArgument(StringArgument<>(selectedUpdateRule_, "", "updateRule", "selects the update rule", possibleUpdateRule.at(0), possibleUpdateRule));
}

template <class IO, class GM, class ACC>
inline void LibDaiFractionalBpCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LibDaiFractionalBp caller" << std::endl;

   if(selectedUpdateRule_ == std::string("PARALL")) {
     parameter_.updateRule_= LibDai_FractionalBp::PARALL;
   }
   else if(selectedUpdateRule_ == std::string("SEQFIX")) {
     parameter_.updateRule_= LibDai_FractionalBp::SEQFIX;
   } 
   else if(selectedUpdateRule_ == std::string("SEQMAX")) {
     parameter_.updateRule_= LibDai_FractionalBp::SEQMAX;
   }
   else if(selectedUpdateRule_ == std::string("SEQRND")) {
     parameter_.updateRule_= LibDai_FractionalBp::SEQRND;
   }
   else {
     throw RuntimeError("Unknown update rule for libdai-bp");
   }

   this-> template infer<LibDai_FractionalBp, TimingVisitorType, typename LibDai_FractionalBp::Parameter>(model, outputfile, verbose, parameter_);

}

template <class IO, class GM, class ACC>
const std::string LibDaiFractionalBpCaller<IO, GM, ACC>::name_ = "LIBDAI-FRACTIONAL-BP";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_FRACTIONAL_BP_CALLER */
