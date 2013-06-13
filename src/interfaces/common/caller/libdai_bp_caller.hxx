#ifndef LIBDAI_BP_CALLER
#define LIBDAI_BP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/bp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiBpCaller : public InferenceCallerBase<IO, GM, ACC, LibDaiBpCaller<IO, GM, ACC> > {
public:
   typedef external::libdai::Bp<GM, ACC> LibDai_Bp;
   typedef InferenceCallerBase<IO, GM, ACC, LibDaiBpCaller<IO, GM, ACC> > BaseClass;
   typedef typename LibDai_Bp::VerboseVisitorType VerboseVisitorType;
   typedef typename LibDai_Bp::EmptyVisitorType EmptyVisitorType;
   typedef typename LibDai_Bp::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   LibDaiBpCaller(IO& ioIn);
   virtual ~LibDaiBpCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename LibDai_Bp::Parameter bpParameter_;
   std::string selectedUpdateRule_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline LibDaiBpCaller<IO, GM, ACC>::LibDaiBpCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LibDaiBpCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(bpParameter_.maxIterations_, "", "maxIt", "Maximum number of iterations.", size_t(bpParameter_.maxIterations_)));
   addArgument(DoubleArgument<>(bpParameter_.tolerance_, "", "bound", "convergence bound.", double(bpParameter_.tolerance_)));
   addArgument(DoubleArgument<>(bpParameter_.damping_, "", "damping", "message damping", double(0.0)));
   addArgument(Size_TArgument<>(bpParameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(bpParameter_.verbose_)));
   std::vector<std::string> possibleUpdateRule;
   possibleUpdateRule.push_back(std::string("PARALL"));
   possibleUpdateRule.push_back(std::string("SEQFIX"));
   possibleUpdateRule.push_back(std::string("SEQRND"));
   possibleUpdateRule.push_back(std::string("SEQMAX"));
   addArgument(StringArgument<>(selectedUpdateRule_, "", "updateRule", "selects the update rule", possibleUpdateRule.at(0), possibleUpdateRule));
}

template <class IO, class GM, class ACC>
inline LibDaiBpCaller<IO, GM, ACC>::~LibDaiBpCaller() {

}

template <class IO, class GM, class ACC>
inline void LibDaiBpCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LibDaiBp caller" << std::endl;

   if(selectedUpdateRule_ == std::string("PARALL")) {
     bpParameter_.updateRule_= LibDai_Bp::PARALL;
   }
   else if(selectedUpdateRule_ == std::string("SEQFIX")) {
     bpParameter_.updateRule_= LibDai_Bp::SEQFIX;
   } 
   else if(selectedUpdateRule_ == std::string("SEQMAX")) {
     bpParameter_.updateRule_= LibDai_Bp::SEQMAX;
   }
   else if(selectedUpdateRule_ == std::string("SEQRND")) {
     bpParameter_.updateRule_= LibDai_Bp::SEQRND;
   }
   else {
     throw RuntimeError("Unknown update rule for libdai-bp");
   }

   this-> template infer<LibDai_Bp, TimingVisitorType, typename LibDai_Bp::Parameter>(model, output, verbose, bpParameter_);
}

template <class IO, class GM, class ACC>
const std::string LibDaiBpCaller<IO, GM, ACC>::name_ = "LIBDAI-BP";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_BP_CALLER */
