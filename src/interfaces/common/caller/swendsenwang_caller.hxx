#ifndef SWENDSENWANG_CALLER_HXX_
#define SWENDSENWANG_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/swendsenwang.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SwendsenWangCaller : public InferenceCallerBase<IO, GM, ACC, SwendsenWangCaller<IO, GM, ACC> > {
public:
   typedef SwendsenWang<GM, ACC> SW;
   typedef InferenceCallerBase<IO, GM, ACC, SwendsenWangCaller<IO, GM, ACC> > BaseClass;
   typedef typename SW::VerboseVisitorType VerboseVisitorType;
   typedef typename SW::EmptyVisitorType EmptyVisitorType;
   typedef typename SW::TimingVisitorType TimingVisitorType;
   const static std::string name_;
   SwendsenWangCaller(IO& ioIn);
   virtual ~SwendsenWangCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename SW::Parameter swParameter_;
};

template <class IO, class GM, class ACC>
inline SwendsenWangCaller<IO, GM, ACC>::SwendsenWangCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of SwendsenWang caller...", ioIn) {
  
   addArgument(Size_TArgument<>(swParameter_.maxNumberOfSamplingSteps_, "", "samplingSteps", "number of sampling steps", (size_t)1000));
   addArgument(Size_TArgument<>(swParameter_.numberOfBurnInSteps_, "", "burninSteps", "number of burnin steps (should always be 0 for optimization)", (size_t)0));
   addArgument(VectorArgument<std::vector<typename GM::LabelType> >(swParameter_.initialState_, "", "startPoint", "location of the file containing a vector which specifies the initial labeling", false));
   addArgument(DoubleArgument<>(swParameter_.lowestAllowedProbability_,"","lowestProb","used to throw an exception if undercut", 1e-6)); 
}

template <class IO, class GM, class ACC>
inline SwendsenWangCaller<IO, GM, ACC>::~SwendsenWangCaller() {

}

template <class IO, class GM, class ACC>
inline void SwendsenWangCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running SwendsenWang caller" << std::endl;

   //check start point
   if(swParameter_.initialState_.size() != model.numberOfVariables()){
      swParameter_.initialState_.resize(model.numberOfVariables(),0);
   }

   this-> template infer<SW, TimingVisitorType, typename SW::Parameter>(model, output, verbose, swParameter_);
}

template <class IO, class GM, class ACC>
const std::string SwendsenWangCaller<IO, GM, ACC>::name_ = "SwendsenWang";

} // namespace interface

} // namespace opengm

#endif /* SwendsenWang_CALLER_HXX_ */
