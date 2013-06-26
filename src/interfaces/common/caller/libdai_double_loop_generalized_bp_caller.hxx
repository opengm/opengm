#ifndef LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_CALLER
#define LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_CALLER

#include <opengm/opengm.hxx>
#include <opengm/inference/external/libdai/double_loop_generalized_bp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class LibDaiDoubleLoopGeneralizedBpCaller : public InferenceCallerBase<IO, GM, ACC, LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC> > {
public:
   typedef external::libdai::DoubleLoopGeneralizedBP<GM, ACC> LibDai_DoubleLoopGeneralizedBP;
   typedef InferenceCallerBase<IO, GM, ACC, LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC> > BaseClass;
   typedef typename LibDai_DoubleLoopGeneralizedBP::VerboseVisitorType VerboseVisitorType;
   typedef typename LibDai_DoubleLoopGeneralizedBP::EmptyVisitorType EmptyVisitorType;
   typedef typename LibDai_DoubleLoopGeneralizedBP::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   LibDaiDoubleLoopGeneralizedBpCaller(IO& ioIn);
   virtual ~LibDaiDoubleLoopGeneralizedBpCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename LibDai_DoubleLoopGeneralizedBP::Parameter parameter_;
   std::string selectedInit_;
   std::string selectedCluster_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);
};

template <class IO, class GM, class ACC>
inline LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC>::LibDaiDoubleLoopGeneralizedBpCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of LibDaiDoubleLoopGeneralizedBpCaller caller...", ioIn) {
   addArgument(Size_TArgument<>(parameter_.loopdepth_, "", "loopDepth", "loopDepth(only needed if heuristic=LOOP )", size_t(parameter_.loopdepth_)));
   addArgument(BoolArgument(parameter_.doubleloop_, "", "doubleLoop", "use double Loop or not"));
   addArgument(Size_TArgument<>(parameter_.maxiter_, "", "maxIt", "Maximum number of iterations.", size_t(parameter_.maxiter_)));
   addArgument(DoubleArgument<>(parameter_.tolerance_, "", "bound", "convergence bound.", double(parameter_.tolerance_)));
   addArgument(Size_TArgument<>(parameter_.verbose_, "", "verboseLevel", "Libdai verbose level", size_t(parameter_.verbose_)));
   std::vector<std::string> possibleInit;
   possibleInit.push_back(std::string("UNIFORM"));
   possibleInit.push_back(std::string("RANDOM"));
   addArgument(StringArgument<>(selectedInit_, "", "initialization", "selects the initialization type", possibleInit.at(0), possibleInit));
   std::vector<std::string> possibleCluster;
   possibleCluster.push_back(std::string("MIN"));
   possibleCluster.push_back(std::string("BETHE"));
   possibleCluster.push_back(std::string("DELTA"));
   possibleCluster.push_back(std::string("LOOP"));
   addArgument(StringArgument<>(selectedCluster_, "", "heuristic", "selects cluster type", possibleCluster.at(0), possibleCluster));
}

template <class IO, class GM, class ACC>
inline LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC>::~LibDaiDoubleLoopGeneralizedBpCaller() {

}

template <class IO, class GM, class ACC>
inline void LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running LibDaiDoubleLoopGeneralizedBP caller" << std::endl;

   if(selectedInit_ == std::string("UNIFORM")) {
     parameter_.init_= LibDai_DoubleLoopGeneralizedBP::UNIFORM;
   }
   else if(selectedInit_ == std::string("RANDOM")) {
     parameter_.init_= LibDai_DoubleLoopGeneralizedBP::RANDOM;
   } 
   else {
     throw RuntimeError("Unknown initialization type for  libdai-double-loop-generaliezed-bp");
   }
   
   if(selectedCluster_ == std::string("MIN")) {
     parameter_.clusters_= LibDai_DoubleLoopGeneralizedBP::MIN;
   }
   else if(selectedCluster_ == std::string("BETHE")) {
     parameter_.clusters_= LibDai_DoubleLoopGeneralizedBP::BETHE;
   }
   else if(selectedCluster_ == std::string("DELTA")) {
     parameter_.clusters_= LibDai_DoubleLoopGeneralizedBP::DELTA;
   } 
   else if(selectedCluster_ == std::string("LOOP")) {
     parameter_.clusters_= LibDai_DoubleLoopGeneralizedBP::LOOP;
   } 
   else {
     throw RuntimeError("Unknown cluster type for libdai-double-loop-generaliezed-bp");
   }

   this-> template infer<LibDai_DoubleLoopGeneralizedBP, TimingVisitorType, typename LibDai_DoubleLoopGeneralizedBP::Parameter>(model, output, verbose, parameter_);
}

template <class IO, class GM, class ACC>
const std::string LibDaiDoubleLoopGeneralizedBpCaller<IO, GM, ACC>::name_ = "LIBDAI-DOUBLE-LOOP-GENERALIZED-BP";

} // namespace interface

} // namespace opengm

#endif /* LIBDAI_DOUBLE_LOOP_GENERALIZED_BP_CALLER */
