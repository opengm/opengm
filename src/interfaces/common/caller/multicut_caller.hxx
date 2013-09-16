#ifndef MULTICUT_CALLER_HXX_
#define MULTICUT_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/multicut.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class MultiCutCaller : public InferenceCallerBase<IO, GM, ACC, MultiCutCaller<IO, GM, ACC> > {
protected:
   typedef Multicut<GM, ACC> MultiCut;
   typedef InferenceCallerBase<IO, GM, ACC, MultiCutCaller<IO, GM, ACC> > BaseClass;
   typedef typename BaseClass::OutputBase OutputBase;
   typedef typename MultiCut::VerboseVisitorType VerboseVisitorType;
   typedef typename MultiCut::EmptyVisitorType EmptyVisitorType;
   typedef typename MultiCut::TimingVisitorType TimingVisitorType;

   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typename MultiCut::Parameter multicutParameter_;
   std::string MWCRoundingType_;
public:
   const static std::string name_;
   MultiCutCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline MultiCutCaller<IO, GM, ACC>::MultiCutCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of MultiCut caller...", ioIn) {
   addArgument(IntArgument<>(multicutParameter_.numThreads_, "", "threads", "number of threads", multicutParameter_.numThreads_));
   addArgument(BoolArgument(multicutParameter_.verbose_, "v", "verbose", "used to activate verbose output"));
   addArgument(DoubleArgument<>(multicutParameter_.cutUp_, "", "cutup", "cut up", multicutParameter_.cutUp_)); 
   addArgument(Size_TArgument<>(multicutParameter_.maximalNumberOfConstraintsPerRound_,"","maxC","maximal number of constraints added per single round",(size_t)1000000));
   //addArgument(BoolArgument(multicutParameter_.addNoneFacetDefiningConstraints_,  "", "nfdc", "add non-facet-defining constraints"));
   //addArgument(DoubleArgument<>(multicutParameter_.violatedThreshold_, "", "vT", "violation threshold", 0.000001));
   addArgument(DoubleArgument<>(multicutParameter_.timeOut_, "", "timeout", "maximal run-time in seconds", 604800.0)); //default=1week
   addArgument(StringArgument<>(multicutParameter_.workFlow_, "", "workflow", "workflow of cutting-plane procedure, e.g. (CC)(ICC)\nSeperation procedures:\n CC    = add violated cycle constraint\n ICC   = add violated integer cycle constraint\n TTC   = add violated terminal triangle constraint\n ITTC  = add violated integer terminal triangle constraint\n MTC   = add violated multi terminal constraint\n IC    = add violated integer constraints\n RIC   = remove inactive constraints\n", false)); //default=1week
   addArgument(DoubleArgument<>(multicutParameter_.edgeRoundingValue_, "", "round", "rounding value for fractional edges", multicutParameter_.edgeRoundingValue_)); 
   std::vector<std::string> permittedMWCRTypes;
   permittedMWCRTypes.push_back("NEAREST");
   permittedMWCRTypes.push_back("DERANDOMIZED");
   permittedMWCRTypes.push_back("PSEUDORANDOMIZED");
   addArgument(StringArgument<>(MWCRoundingType_, "", "MWCRounding", "select rounding-method in the labelsimplex", permittedMWCRTypes.at(0), permittedMWCRTypes));
   addArgument(Size_TArgument<>(multicutParameter_.reductionMode_,"","reductionMode","Higher order reduction mode (1,2, or 3)",(size_t)3));
}

template <class IO, class GM, class ACC>
inline void MultiCutCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running MultiCut caller" << std::endl;

    //LabelInitialType
   if(MWCRoundingType_ == "NEAREST") {
      multicutParameter_.MWCRounding_ =  MultiCut::Parameter::NEAREST;
   } else if(MWCRoundingType_ == "DERANDOMIZED") {
      multicutParameter_.MWCRounding_ =  MultiCut::Parameter::DERANDOMIZED;
   } else if(MWCRoundingType_ == "PSEUDORANDOMIZED") {
      multicutParameter_.MWCRounding_ =  MultiCut::Parameter::PSEUDODERANDOMIZED;
   } else {
      throw RuntimeError("Unknown rounding type for multiway cut!");
   }


   this-> template infer<MultiCut, TimingVisitorType, typename MultiCut::Parameter>(model, output, verbose, multicutParameter_);
}

template <class IO, class GM, class ACC>
const std::string MultiCutCaller<IO, GM, ACC>::name_ = "MULTICUT";

} // namespace interface

} // namespace opengm

#endif /* MULTICUT_CALLER_HXX_ */
