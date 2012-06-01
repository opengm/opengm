#ifndef BRUTEFORCE_CALLER_HXX_
#define BRUTEFORCE_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/bruteforce.hxx>

#include "inference_caller_base.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class BruteforceCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   typedef typename opengm::Bruteforce<GM, ACC> Bruteforce;
   typename Bruteforce::Parameter bruteforceParameter_;
   typedef typename Bruteforce::VerboseVisitorType VerboseVisitorType;
   typedef typename Bruteforce::EmptyVisitorType EmptyVisitorType;
   typedef typename Bruteforce::TimingVisitorType TimingVisitorType;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
public:
   const static std::string name_;
   BruteforceCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline BruteforceCaller<IO, GM, ACC>::BruteforceCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>("Bruteforce", "detailed description of Bruteforce Parser...", ioIn){

}

template <class IO, class GM, class ACC>
inline void BruteforceCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running Bruteforce caller" << std::endl;

   this-> template infer<Bruteforce, TimingVisitorType, typename Bruteforce::Parameter>(model, outputfile, verbose, bruteforceParameter_);
}

template <class IO, class GM, class ACC>
const std::string BruteforceCaller<IO, GM, ACC>::name_ = "BRUTEFORCE";

} // namespace interface

} // namespace opengm

#endif /* BRUTEFORCE_CALLER_HXX_ */
