#ifndef ICM_CALLER_HXX_
#define ICM_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/icm.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class ICMCaller : public InferenceCallerBase<IO, GM, ACC, ICMCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::ICM<GM, ACC> ICM;
   typedef InferenceCallerBase<IO, GM, ACC, ICMCaller<IO, GM, ACC> > BaseClass;
   typedef typename ICM::VerboseVisitorType VerboseVisitorType;
   typedef typename ICM::EmptyVisitorType EmptyVisitorType;
   typedef typename ICM::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   ICMCaller(IO& ioIn);
   virtual ~ICMCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename ICM::Parameter icmParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline ICMCaller<IO, GM, ACC>::ICMCaller(IO& ioIn)
   : BaseClass("ICM", "detailed description of ICM Parser...", ioIn) {
   addArgument(VectorArgument<std::vector<typename ICM::LabelType> >(icmParameter_.startPoint_, "x0", "startingpoint", "location of the file containing the values for the starting point", false));
}

template <class IO, class GM, class ACC>
inline ICMCaller<IO, GM, ACC>::~ICMCaller() {

}

template <class IO, class GM, class ACC>
inline void ICMCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running ICM caller" << std::endl;

   this-> template infer<ICM, TimingVisitorType, typename ICM::Parameter>(model, output, verbose, icmParameter_);
}

template <class IO, class GM, class ACC>
const std::string ICMCaller<IO, GM, ACC>::name_ = "ICM";

} // namespace interface

} // namespace opengm

#endif /* ICM_CALLER_HXX_ */
