#ifndef BRUTEFORCE_CALLER_HXX_
#define BRUTEFORCE_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/bruteforce.hxx>

#include "inference_caller_base.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class BruteforceCaller : public InferenceCallerBase<IO, GM, ACC, BruteforceCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::Bruteforce<GM, ACC> Bruteforce;
   typedef InferenceCallerBase<IO, GM, ACC, BruteforceCaller<IO, GM, ACC> > BaseClass;
   typedef typename Bruteforce::VerboseVisitorType VerboseVisitorType;
   typedef typename Bruteforce::EmptyVisitorType EmptyVisitorType;
   typedef typename Bruteforce::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   BruteforceCaller(IO& ioIn);
   virtual ~BruteforceCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename Bruteforce::Parameter bruteforceParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline BruteforceCaller<IO, GM, ACC>::BruteforceCaller(IO& ioIn)
   : BaseClass("Bruteforce", "detailed description of Bruteforce Parser...", ioIn){

}

template <class IO, class GM, class ACC>
inline BruteforceCaller<IO, GM, ACC>::~BruteforceCaller() {

}

template <class IO, class GM, class ACC>
inline void BruteforceCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Bruteforce caller" << std::endl;

   this-> template infer<Bruteforce, TimingVisitorType, typename Bruteforce::Parameter>(model, output, verbose, bruteforceParameter_);
}

template <class IO, class GM, class ACC>
const std::string BruteforceCaller<IO, GM, ACC>::name_ = "BRUTEFORCE";

} // namespace interface

} // namespace opengm

#endif /* BRUTEFORCE_CALLER_HXX_ */
