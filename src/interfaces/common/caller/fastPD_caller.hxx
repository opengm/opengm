#ifndef FASTPD_CALLER_HXX_
#define FASTPD_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/fastPD.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class FastPDCaller : public InferenceCallerBase<IO, GM, ACC, FastPDCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::external::FastPD<GM> FastPD; 
   typedef InferenceCallerBase<IO, GM, ACC, FastPDCaller<IO, GM, ACC> > BaseClass;
   typedef typename FastPD::VerboseVisitorType VerboseVisitorType;
   typedef typename FastPD::EmptyVisitorType EmptyVisitorType;
   typedef typename FastPD::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   FastPDCaller(IO& ioIn);
   virtual ~FastPDCaller();
protected:
   typename FastPD::Parameter fastPDParameter_;

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer; 

   typedef typename BaseClass::OutputBase OutputBase;
   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline FastPDCaller<IO, GM, ACC>::FastPDCaller(IO& ioIn)
   : BaseClass("FASTPD", "detailed description of FastPD Parser...", ioIn){
   addArgument(Size_TArgument<>(fastPDParameter_.numberOfIterations_, "", "maxIt", "Maximum number of iterations.", fastPDParameter_.numberOfIterations_));
}

template <class IO, class GM, class ACC>
inline FastPDCaller<IO, GM, ACC>::~FastPDCaller() {

}

template <class IO, class GM, class ACC>
inline void FastPDCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running FastPD caller" << std::endl;

   this-> template infer<FastPD, TimingVisitorType, typename FastPD::Parameter>(model, output, verbose, fastPDParameter_);
}

template <class IO, class GM, class ACC>
const std::string FastPDCaller<IO, GM, ACC>::name_ = "FASTPD";

} // namespace interface

} // namespace opengm

#endif /* FASTPD_CALLER_HXX_ */
