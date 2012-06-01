#ifndef SAT_CALLER_HXX_
#define SAT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/sat.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class SATCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:

   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef SAT<GM> S_A_T;
   typedef typename S_A_T::VerboseVisitorType VerboseVisitorType;
   typedef typename S_A_T::EmptyVisitorType EmptyVisitorType;
   typedef typename S_A_T::TimingVisitorType TimingVisitorType;
   typename S_A_T::Parameter satParameter_;
public:
   const static std::string name_;
   SATCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline SATCaller<IO, GM, ACC>::SATCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of SAT caller...", ioIn) {
}

template <class IO, class GM, class ACC>
inline void SATCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running SAT caller" << std::endl;

   this-> template infer<S_A_T, TimingVisitorType, typename S_A_T::Parameter>(model, outputfile, verbose, satParameter_);
/*   S_A_T sat(model, satParameter_);

   std::vector<typename S_A_T::ValueType> states;
   std::cout << "Inferring!" << std::endl;
   if(!(sat.infer() == NORMAL)) {
      std::string error("SAT did not solve the problem.");
      io_.errorStream() << error << std::endl;
      throw RuntimeError(error);
   }
   std::cout << "writing states in vector!" << std::endl;
   states.push_back(sat.value());
   storeVector(outputfile, states);*/
}

template <class IO, class GM, class ACC>
const std::string SATCaller<IO, GM, ACC>::name_ = "SAT";

} // namespace interface

} // namespace opengm

#endif /* SAT_CALLER_HXX_ */
