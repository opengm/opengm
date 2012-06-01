#ifndef FASTINF_BP_CALLER
#define FASTINF_BP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/fastinf/bp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class FastInfBpCaller : public InferenceCallerBase<IO, GM, ACC> {
protected:
   using InferenceCallerBase<IO, GM, ACC>::addArgument;
   using InferenceCallerBase<IO, GM, ACC>::io_;
   using InferenceCallerBase<IO, GM, ACC>::infer;
   virtual void runImpl(GM& model, StringArgument<>& outputfile, const bool verbose);
   typedef external::fastinf::Bp<GM, ACC> FastInf_Bp;
   typedef typename FastInf_Bp::VerboseVisitorType VerboseVisitorType;
   typedef typename FastInf_Bp::EmptyVisitorType EmptyVisitorType;
   typedef typename FastInf_Bp::TimingVisitorType TimingVisitorType;
   typename FastInf_Bp::Parameter bpParameter_;
   std::string selectedQueue_;
public:
   const static std::string name_;
   FastInfBpCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline FastInfBpCaller<IO, GM, ACC>::FastInfBpCaller(IO& ioIn)
   : InferenceCallerBase<IO, GM, ACC>(name_, "detailed description of FastInfBpCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(bpParameter_.maxMessages_, "", "maxIt", "Maximum number of iterations.", size_t(bpParameter_.maxMessages_)));
   addArgument(DoubleArgument<>(bpParameter_.threshold_, "", "bound", "convergence bound.", double(bpParameter_.threshold_)));
   addArgument(DoubleArgument<>(bpParameter_.damping_, "", "damping", "message damping", double(0.0)));
   addArgument(Size_TArgument<>(bpParameter_.maxSeconds_, "", "maxTime", "Maximum time (in sek.) allowed for inference.", size_t(bpParameter_.maxSeconds_)));
   addArgument(BoolArgument(bpParameter_.functionSharing_, "", "shareFunctions", "Allow functions to be shared"));
   std::vector<std::string> possibleQueues;
   possibleQueues.push_back(std::string("UNWEIGHTED"));
   possibleQueues.push_back(std::string("WEIGHTED"));
   addArgument(StringArgument<>(selectedQueue_, "", "queueType", "selects the type of the message", possibleQueues.at(0), possibleQueues));
}

template <class IO, class GM, class ACC>
inline void FastInfBpCaller<IO, GM, ACC>::runImpl(GM& model, StringArgument<>& outputfile, const bool verbose) {
   std::cout << "running FastInfBp caller" << std::endl;

   if(selectedQueue_ == std::string("UNWEIGHTED")) {
     bpParameter_.queueType_= FastInf_Bp::UNWEIGHTED;
   }
   else if(selectedQueue_ == std::string("WEIGHTED")) {
     bpParameter_.queueType_= FastInf_Bp::WEIGHTED;
   } 
   else {
     throw RuntimeError("Unknown queueType for fastinf-bp");
   }

   this-> template infer<FastInf_Bp, TimingVisitorType, typename FastInf_Bp::Parameter>(model, outputfile, verbose, bpParameter_);

}

template <class IO, class GM, class ACC>
const std::string FastInfBpCaller<IO, GM, ACC>::name_ = "FASTINF-BP";

} // namespace interface

} // namespace opengm

#endif /* FASTINF_BP_CALLER */
