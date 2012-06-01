#ifndef LOC_CALLER_HXX_
#define LOC_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/loc.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LOCCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef LOC<GM, ACC> L_O_C;
   typedef typename L_O_C::VerboseVisitorType VerboseVisitorType;
   typedef typename L_O_C::EmptyVisitorType EmptyVisitorType;
   typedef typename L_O_C::TimingVisitorType TimingVisitorType;
   typename L_O_C::Parameter locParameter_;
public:
   const static std::string name_;
   LOCCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LOCCaller<IO, GM, ACC>::LOCCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of LOC caller...", ioIn) {
   addArgument(DoubleArgument<>(locParameter_.phi_, "", "phi", "phi", locParameter_.phi_));
   addArgument(Size_TArgument<>(locParameter_.maxRadius_, "", "maxr", "maximum radius", locParameter_.maxRadius_));
   addArgument(Size_TArgument<>(locParameter_.maxIterations_, "", "maxIt", "Maximum number of iterations.", locParameter_.maxIterations_));
   addArgument(VectorArgument<std::vector<size_t> >(locParameter_.startPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline void LOCCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running LOC caller" << std::endl;

   this-> template infer<L_O_C, TimingVisitorType, typename L_O_C::Parameter>(model, outputfile, verbose, locParameter_);
/*   L_O_C loc(model, locParameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(loc.infer() == NORMAL)) {
      std::string error("LOC did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(loc.arg(states) == NORMAL)) {
      std::string error("LOC could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string LOCCaller<IO, GM, ACC>::name_ = "LOC";

} // namespace interface

} // namespace opengm

#endif /* LOC_CALLER_HXX_ */
