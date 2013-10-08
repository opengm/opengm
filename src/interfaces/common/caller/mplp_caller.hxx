#ifndef OPENGM_EXTERNAL_MPLP_CALLER_HXX_
#define OPENGM_EXTERNAL_MPLP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/external/mplp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MPLPCaller : public InferenceCallerBase<IO, GM, ACC, MPLPCaller<IO, GM, ACC> > {
public:
   typedef typename opengm::external::MPLP<GM> MPLP;
   typedef InferenceCallerBase<IO, GM, ACC, MPLPCaller<IO, GM, ACC> > BaseClass;
   typedef typename MPLP::VerboseVisitorType VerboseVisitorType;
   typedef typename MPLP::EmptyVisitorType EmptyVisitorType;
   typedef typename MPLP::TimingVisitorType TimingVisitorType;

   const static std::string name_;
   MPLPCaller(IO& ioIn);
   virtual ~MPLPCaller();
protected:
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   typedef typename BaseClass::OutputBase OutputBase;

   typename MPLP::Parameter mplpParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

};

template <class IO, class GM, class ACC>
inline MPLPCaller<IO, GM, ACC>::MPLPCaller(IO& ioIn)
   : BaseClass("MPLP", "detailed description of MPLP Parser...", ioIn) {
   addArgument(Size_TArgument<>(mplpParameter_.maxIterTight_, "", "maxIterTight", "Maximum number of tightening iterations"      , mplpParameter_.maxIterTight_));
   addArgument(Size_TArgument<>(mplpParameter_.maxIterLP_,    "", "maxIterLP",    "Number of initial LP iterations"              , mplpParameter_.maxIterLP_));
   addArgument(Size_TArgument<>(mplpParameter_.maxIterLater_, "", "maxIterLater", "Number of LP iterations after each tightening", mplpParameter_.maxIterLater_));

   addArgument(Size_TArgument<>(mplpParameter_.numClusToAddMin_, "", "numClusToAddMin", "Minimum number of clusters to add", mplpParameter_.numClusToAddMin_));
   addArgument(Size_TArgument<>(mplpParameter_.numClusToAddMax_, "", "numClusToAddMax", "Maximum number of clusters to add", mplpParameter_.numClusToAddMax_));

   addArgument(DoubleArgument<>(mplpParameter_.objDelThr_, "", "objDelThr", "objDelThr", mplpParameter_.objDelThr_));
   addArgument(DoubleArgument<>(mplpParameter_.intGapThr_, "", "intGapThr", "intGapThr", mplpParameter_.intGapThr_));

   addArgument(BoolArgument(mplpParameter_.UAIsettings_,          "", "UAIsettings", "Use settings for UAI inference competition"));
   addArgument(BoolArgument(mplpParameter_.addEdgeIntersections_, "", "addEdgeIntersections", "Add edge intersections"));
   addArgument(BoolArgument(mplpParameter_.doGlobalDecoding_,     "", "doGlobalDecoding", "Do global decoding"));
   addArgument(BoolArgument(mplpParameter_.useDecimation_,        "", "useDecimation", "Use decimation"));
   addArgument(BoolArgument(mplpParameter_.lookForCSPs_,          "", "lookForCSPs", "Look for CSPs"));

   addArgument(DoubleArgument<>(mplpParameter_.maxTime_,   "", "maxTime"  , "Total number of seconds allowed to run"               , mplpParameter_.maxTime_));
   addArgument(DoubleArgument<>(mplpParameter_.maxTimeLP_, "", "maxTimeLP", "Total number of seconds allowed to run for initial LP", mplpParameter_.maxTimeLP_));

   addArgument(StringArgument<>(mplpParameter_.logFile_,      "", "logFile", "Path to MPLP log output file"));
// addArgument(StringArgument<>(mplpParameter_.inputFile_,    "", "inputFile", "Path to MPLP input file (if inputFile is set, the MPLP-Algorithm will load the model from the given file instead of using the openGM model)"));
// addArgument(StringArgument<>(mplpParameter_.evidenceFile_, "", "evidenceFile", "Path to MPLP evidence file (will only be used if inputFile is set"));

   addArgument(IntArgument<>(mplpParameter_.seed_, "", "seed", "Seed for random number generator", mplpParameter_.seed_));
}

template <class IO, class GM, class ACC>
inline MPLPCaller<IO, GM, ACC>::~MPLPCaller() {

}

template <class IO, class GM, class ACC>
inline void MPLPCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running MPLP caller" << std::endl;

   this-> template infer<MPLP, TimingVisitorType, typename MPLP::Parameter>(model, output, verbose, mplpParameter_);
}

template <class IO, class GM, class ACC>
const std::string MPLPCaller<IO, GM, ACC>::name_ = "MPLP";

} // namespace interface

} // namespace opengm


#endif /* OPENGM_EXTERNAL_MPLP_CALLER_HXX_ */
