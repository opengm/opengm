#ifndef LAZYFLIPPER_CALLER_HXX_
#define LAZYFLIPPER_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/lazyflipper.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LazyFlipperCaller : public InferenceCallerBase<IO, GM, ACC, LazyFlipperCaller<IO, GM, ACC> > {
public:
   typedef InferenceCallerBase<IO, GM, ACC, LazyFlipperCaller<IO, GM, ACC> > BaseClass;
   typedef LazyFlipper<GM, ACC> Lazy_Flipper;
   typedef typename Lazy_Flipper::VerboseVisitorType VerboseVisitorType;
   typedef typename Lazy_Flipper::EmptyVisitorType EmptyVisitorType;
   typedef typename Lazy_Flipper::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LazyFlipperCaller(IO& ioIn);
   virtual ~LazyFlipperCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   typedef typename BaseClass::OutputBase OutputBase;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename Lazy_Flipper::Parameter lazyflipperParameter_;
};

template <class IO, class GM, class ACC>
inline LazyFlipperCaller<IO, GM, ACC>::LazyFlipperCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of Lazy Flipper caller...", ioIn) {
   addArgument(Size_TArgument<>(lazyflipperParameter_.maxSubgraphSize_, "", "maxsize", "maximum sub-graph size", lazyflipperParameter_.maxSubgraphSize_));
   addArgument(VectorArgument<std::vector<typename Lazy_Flipper::LabelType> >(lazyflipperParameter_.startingPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline LazyFlipperCaller<IO, GM, ACC>::~LazyFlipperCaller() {

}

template <class IO, class GM, class ACC>
inline void LazyFlipperCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Lazy Flipper caller" << std::endl;

   this-> template infer<Lazy_Flipper, TimingVisitorType, typename Lazy_Flipper::Parameter>(model, output, verbose, lazyflipperParameter_);
}

template <class IO, class GM, class ACC>
const std::string LazyFlipperCaller<IO, GM, ACC>::name_ = "LAZYFLIPPER";

} // namespace interface

} // namespace opengm

#endif /* LAZYFLIPPER_CALLER_HXX_ */
