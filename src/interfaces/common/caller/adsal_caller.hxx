#ifndef ADSAL_CALLER_HXX_
#define ADSAL_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/trws/trws_adsal.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class ADSalCaller : public InferenceCallerBase<IO, GM, ACC, ADSalCaller<IO, GM, ACC> > {
protected:
   typedef InferenceCallerBase<IO, GM, ACC, ADSalCaller<IO, GM, ACC> > BaseClass;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typedef ADSal<GM, ACC> ADSalType;
   typedef typename ADSalType::VerboseVisitorType VerboseVisitorType;
   typedef typename ADSalType::EmptyVisitorType EmptyVisitorType;
   typedef typename ADSalType::TimingVisitorType TimingVisitorType;
   typename ADSalType::Parameter adsalParameter_;

   size_t relativePrecision;
   std::string stringDecompositionType;
   std::string smoothingStrategyType;
   size_t lazyLPPrimalBoundComputation;
   size_t lazyDerivativeComputation;
   double startSmoothingValue;
   size_t slowComputations;
   size_t noNormalization;
   double precision;
public:
   const static std::string name_;
   ADSalCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline ADSalCaller<IO, GM, ACC>::ADSalCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of the internal ADSal caller...", ioIn),
     startSmoothingValue(adsalParameter_.startSmoothingValue())
    {
	std::vector<size_t> boolVec(2); boolVec[0]=0; boolVec[1]=1;
	std::vector<std::string> stringVec(3); stringVec[0]="GENERAL"; stringVec[1]="GRID"; stringVec[2]="EDGE";
	std::vector<std::string> stringVecSmoothStrategy(4); stringVecSmoothStrategy[0]="ADAPTIVE_DIMINISHING"; stringVecSmoothStrategy[1]="WC_DIMINISHING";stringVecSmoothStrategy[2]="ADAPTIVE_PRECISIONORIENTED"; stringVecSmoothStrategy[3]="WC_PRECISIONORIENTED";
	addArgument(Size_TArgument<>(adsalParameter_.maxNumberOfIterations(), "", "maxIt", "Maximum number of iterations.",true));
	addArgument(DoubleArgument<>(precision, "", "precision", "Duality gap based absolute precision to be obtained. Default is 0.0. Use parameter --relative to select the relative one",(double)0.0));
	addArgument(Size_TArgument<>(relativePrecision, "", "relative", "If set to 1 , then the parameter --precision determines a relative precision value. Default is an absolute one",(size_t)0,boolVec));
	addArgument(StringArgument<>(stringDecompositionType, "d", "decomposition", "Select decomposition: GENERAL, GRID or EDGE. Default is GENERAL", false,stringVec));
	addArgument(Size_TArgument<>(adsalParameter_.numberOfInternalIterations(), "", "numberOfInternalIterations", "Number of internal iterations (between changes of smoothing).",false));
	addArgument(DoubleArgument<>(adsalParameter_.smoothingGapRatio(), "", "smoothingGapRatio", "Constant gamma - smoothing gap ratio",false));
	addArgument(Size_TArgument<>(lazyLPPrimalBoundComputation, "", "lazyLPPrimalBoundComputation", "If set to 1 the fractal primal bound is not computed when the primal bound improved over the last outer iteration",(size_t)0,boolVec));
    addArgument(Size_TArgument<>(lazyDerivativeComputation, "", "lazyDerivativeComputation", "If set to 1 the derivative w.r.t. smoothing is computed only when the fractal primal bound computation is needed",(size_t)0,boolVec));
	addArgument(DoubleArgument<>(adsalParameter_.smoothingDecayMultiplier(), "", "smoothingDecayMultiplier", "Smoothing decay parameter defines the rate of a forced smoothing decay. Default is <= 0, which switches off the forced decay",false));
	addArgument(DoubleArgument<>(startSmoothingValue, "", "ADSal_startSmoothing", "ADSal::Starting smoothing value. Default is automatic selection",false));
	addArgument(Size_TArgument<>(slowComputations, "", "slowComputations", "If set to 1 the type of the pairwise factors (Potts for now, will be extended to truncated linear and quadratic) will NOT be used to speed up computations of the TRWS subsolver.",(size_t)0,boolVec));
    addArgument(Size_TArgument<>(noNormalization, "", "noNormalization", "If set to 1 the canonical normalization is NOT used in the TRWS subsolver.",(size_t)0,boolVec));
	addArgument(Size_TArgument<>(adsalParameter_.maxNumberOfPresolveIterations(), "", "maxNumberOfPresolveIterations", "The number of TRWS iterations used a a presolver of the ADSal algorithm",false));
	addArgument(DoubleArgument<>(adsalParameter_.presolveMinRelativeDualImprovement(), "", "presolveMinRelativeDualImprovement", "The minimal improvement of the dual function in presolver. If the actual improvement is less, it stops the presolver",false));
	addArgument(Size_TArgument<>(adsalParameter_.maxPrimalBoundIterationNumber(), "", "maxPrimalBoundIterationNumber", "The maximal iteration number of the transportation solver for estimating the primal fractal solution",false));
	addArgument(DoubleArgument<>(adsalParameter_.primalBoundRelativePrecision(), "", "primalBoundRelativePrecision", "The relative precision used to solve the transportation problem for estimating the primal fractal solution",false));
	addArgument(BoolArgument(adsalParameter_.verbose(), "", "debugverbose", "If set the solver will output debug information to the stdout"));
	addArgument(StringArgument<>(smoothingStrategyType, "", "smoothingStrategy", "Select smoothing strategy: ADAPTIVE_DIMINISHING, WC_DIMINISHING, ADAPTIVE_PRECISIONORIENTED or WC_PRECISIONORIENTED. Default is ADAPTIVE_DIMINISHING", false,stringVecSmoothStrategy));
}

template <class IO, class GM, class ACC>
inline void ADSalCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running ADSal caller" << std::endl;
   adsalParameter_.isAbsolutePrecision()=(relativePrecision==0);
   adsalParameter_.decompositionType()=trws_base::DecompositionStorage<GM>::getStructureType(stringDecompositionType);
   adsalParameter_.setStartSmoothingValue(startSmoothingValue);
   adsalParameter_.lazyLPPrimalBoundComputation()=(lazyLPPrimalBoundComputation==1);
   adsalParameter_.lazyDerivativeComputation()=(lazyDerivativeComputation==1);
   adsalParameter_.setFastComputations((slowComputations==0));
   adsalParameter_.setCanonicalNormalization((noNormalization==0));
   adsalParameter_.setPrecision(precision);
   adsalParameter_.smoothingStrategy()=ADSalType::Parameter::SmoothingParametersType::getSmoothingStrategyType(smoothingStrategyType);

   this-> template infer<ADSalType, TimingVisitorType, typename ADSalType::Parameter>(model, output, verbose, adsalParameter_);
}

template <class IO, class GM, class ACC>
const std::string ADSalCaller<IO, GM, ACC>::name_ = "ADSal";

} // namespace interface

} // namespace opengm

#endif /* ADSAL_CALLER_HXX_ */
