#ifndef LAZYFLIPPER_CALLER_HXX_
#define LAZYFLIPPER_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lazyflipper.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LazyFlipperCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef LazyFlipper<GM, ACC> Lazy_Flipper;
   typedef typename Lazy_Flipper::VerboseVisitorType VerboseVisitorType;
   typedef typename Lazy_Flipper::EmptyVisitorType EmptyVisitorType;
   typedef typename Lazy_Flipper::TimingVisitorType TimingVisitorType;
   typename Lazy_Flipper::Parameter lazyflipperParameter_;
public:
   const static std::string name_;
   LazyFlipperCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline LazyFlipperCaller<IO, GM, ACC>::LazyFlipperCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of Lazy Flipper caller...", ioIn) {
   addArgument(Size_TArgument<>(lazyflipperParameter_.maxSubgraphSize_, "", "maxsize", "maximum sub-graph size", lazyflipperParameter_.maxSubgraphSize_));
   addArgument(VectorArgument<std::vector<typename Lazy_Flipper::LabelType> >(lazyflipperParameter_.startingPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline void LazyFlipperCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running Lazy Flipper caller" << std::endl;

   this-> template infer<Lazy_Flipper, TimingVisitorType, typename Lazy_Flipper::Parameter>(model, outputfile, verbose, lazyflipperParameter_);

/*   Lazy_Flipper lazyflipper(model, lazyflipperParameter_);

   std::vector<size_t> states;
   std::cout << "Inferring!" << std::endl;
   if(!(lazyflipper.infer() == NORMAL)) {
      std::string error("Lazy Flipper did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   if(!(lazyflipper.arg(states) == NORMAL)) {
      std::string error("Lazy Flipper could not return optimal argument.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }

   io_.read(outputfile);
   io_.storeVector(outputfile.getValue(), states);*/
}

template <class IO, class GM, class ACC>
const std::string LazyFlipperCaller<IO, GM, ACC>::name_ = "LAZYFLIPPER";

} // namespace interface

} // namespace opengm

#endif /* LAZYFLIPPER_CALLER_HXX_ */
