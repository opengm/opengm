#ifndef SAT_CALLER_HXX_
#define SAT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/sat.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SATCaller : public InferenceCallerBase<IO, GM, ACC, SATCaller<IO, GM, ACC> > {
public:
   typedef SAT<GM> S_A_T;
   typedef InferenceCallerBase<IO, GM, ACC, SATCaller<IO, GM, ACC> > BaseClass;
   typedef typename S_A_T::VerboseVisitorType VerboseVisitorType;
   typedef typename S_A_T::EmptyVisitorType EmptyVisitorType;
   typedef typename S_A_T::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   SATCaller(IO& ioIn);
   virtual ~SATCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename S_A_T::Parameter satParameter_;
};

template <class IO, class GM, class ACC>
inline SATCaller<IO, GM, ACC>::SATCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of SAT caller...", ioIn) {
}

template <class IO, class GM, class ACC>
inline SATCaller<IO, GM, ACC>::~SATCaller() {

}

template <class IO, class GM, class ACC>
inline void SATCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running SAT caller" << std::endl;

   this-> template infer<S_A_T, TimingVisitorType, typename S_A_T::Parameter>(model, output, verbose, satParameter_);
}

template <class IO, class GM, class ACC>
const std::string SATCaller<IO, GM, ACC>::name_ = "SAT";

} // namespace interface

} // namespace opengm

#endif /* SAT_CALLER_HXX_ */
