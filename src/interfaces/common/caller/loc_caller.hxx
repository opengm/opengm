#ifndef LOC_CALLER_HXX_
#define LOC_CALLER_HXX_
#ifdef WITH_AD3
#include <opengm/opengm.hxx>
#include <opengm/inference/loc.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LOCCaller : public InferenceCallerBase<IO, GM, ACC, LOCCaller<IO, GM, ACC> > {
public:
   typedef InferenceCallerBase<IO, GM, ACC, LOCCaller<IO, GM, ACC> > BaseClass;
   typedef LOC<GM, ACC> L_O_C;
   typedef typename L_O_C::VerboseVisitorType VerboseVisitorType;
   typedef typename L_O_C::EmptyVisitorType EmptyVisitorType;
   typedef typename L_O_C::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   LOCCaller(IO& ioIn);
   virtual ~LOCCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename L_O_C::Parameter locParameter_;
};

template <class IO, class GM, class ACC>
inline LOCCaller<IO, GM, ACC>::LOCCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LOC caller...", ioIn) {
   addArgument(DoubleArgument<>(locParameter_.phi_, "", "phi", "phi", locParameter_.phi_));
   addArgument(Size_TArgument<>(locParameter_.maxRadius_, "", "maxr", "maximum radius", locParameter_.maxRadius_));
   addArgument(Size_TArgument<>(locParameter_.maxIterations_, "", "maxIt", "Maximum number of iterations.", locParameter_.maxIterations_));

   addArgument(Size_TArgument<>(locParameter_.stopAfterNBadIterations_,"",
      "autoStop","stop after n iterations without improvement (0 means use gm.numberOfVariables)"));
   //addArgument(VectorArgument<std::vector<typename L_O_C::LabelType> >(locParameter_.startPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline LOCCaller<IO, GM, ACC>::~LOCCaller() {

}

template <class IO, class GM, class ACC>
inline void LOCCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LOC caller" << std::endl;

   this-> template infer<L_O_C, TimingVisitorType, typename L_O_C::Parameter>(model, output, verbose, locParameter_);
}

template <class IO, class GM, class ACC>
const std::string LOCCaller<IO, GM, ACC>::name_ = "LOC";

} // namespace interface

} // namespace opengm

#endif
#endif /* LOC_CALLER_HXX_ */
