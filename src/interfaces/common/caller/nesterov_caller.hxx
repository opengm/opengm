#ifndef NESTEROV_CALLER_HXX_
#define NESTEROV_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/trws/smooth_nesterov.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"


namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class NesterovCaller : public InferenceCallerBase<IO, GM, ACC, NesterovCaller<IO, GM, ACC> > {
protected:
   typedef InferenceCallerBase<IO, GM, ACC, NesterovCaller<IO, GM, ACC> > BaseClass;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   typedef NesterovAcceleratedGradient<GM, ACC> NesterovType;
   typedef typename NesterovType::VerboseVisitorType VerboseVisitorType;
   typedef typename NesterovType::EmptyVisitorType EmptyVisitorType;
   typedef typename NesterovType::TimingVisitorType TimingVisitorType;
   typename NesterovType::Parameter nesterovParameter_;

   size_t relativePrecision;
   std::string stringDecompositionType;
   std::string smoothingStrategyType;
   std::string gradientStepType;
   size_t lazyLPPrimalBoundComputation;
   //size_t lazyDerivativeComputation;
   double startSmoothingValue;
   size_t slowComputations;
   size_t noNormalization;
   double precision;
public:
   const static std::string name_;
   NesterovCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline NesterovCaller<IO, GM, ACC>::NesterovCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of the internal Nesterov caller...", ioIn),
     startSmoothingValue(nesterovParameter_.startSmoothingValue())
    {
	std::vector<size_t> boolVec(2); boolVec[0]=0; boolVec[1]=1;
	std::vector<std::string> stringVec(3); stringVec[0]="GENERAL"; stringVec[1]="GRID";  stringVec[2]="EDGE";
	std::vector<std::string> stringVecSmoothStrategy(4); stringVecSmoothStrategy[0]="ADAPTIVE_DIMINISHING"; stringVecSmoothStrategy[1]="WC_DIMINISHING";
	stringVecSmoothStrategy[2]="ADAPTIVE_PRECISIONORIENTED"; stringVecSmoothStrategy[3]="WC_PRECISIONORIENTED";
	addArgument(Size_TArgument<>(nesterovParameter_.maxNumberOfIterations(), "", "maxIt", "Maximum number of iterations.",true));
	addArgument(DoubleArgument<>(precision, "", "precision", "Duality gap based absolute precision to be obtained. Default is 0.0. Use parameter --relative to select the relative one",(double)0.0));
	addArgument(Size_TArgument<>(relativePrecision, "", "relative", "If set to 1 , then the parameter --precision determines a relative precision value. Default is an absolute one",(size_t)0,boolVec));
	addArgument(StringArgument<>(stringDecompositionType, "d", "decomposition", "Select decomposition: GENERAL, GRID or EDGE. Default is GENERAL", false,stringVec));
	addArgument(Size_TArgument<>(nesterovParameter_.numberOfInternalIterations(), "", "numberOfInternalIterations", "Number of internal iterations (between changes of smoothing).",false));
	addArgument(DoubleArgument<>(nesterovParameter_.smoothingGapRatio(), "", "smoothingGapRatio", "Constant gamma - smoothing gap ratio",false));
	addArgument(Size_TArgument<>(lazyLPPrimalBoundComputation, "", "lazyLPPrimalBoundComputation", "If set to 1 the fractal primal bound is not computed when the primal bound improved over the last outer iteration",(size_t)0,boolVec));
    //addArgument(Size_TArgument<>(lazyDerivativeComputation, "", "lazyDerivativeComputation", "If set to 1 the derivative w.r.t. smoothing is computed only when the fractal primal bound computation is needed",(size_t)0,boolVec));
	addArgument(DoubleArgument<>(nesterovParameter_.smoothingDecayMultiplier(), "", "smoothingDecayMultiplier", "Smoothing decay parameter defines the rate of a forced smoothing decay. Default is <= 0, which switches off the forced decay",false));
	addArgument(DoubleArgument<>(startSmoothingValue, "", "Nesterov_startSmoothing", "Nesterov::Starting smoothing value. Default is automatic selection",false));
	addArgument(Size_TArgument<>(slowComputations, "", "slowComputations", "If set to 1 the type of the pairwise factors (Potts for now, will be extended to truncated linear and quadratic) will NOT be used to speed up computations of the TRWS subsolver.",(size_t)0,boolVec));
    addArgument(Size_TArgument<>(noNormalization, "", "noNormalization", "If set to 1 the canonical normalization is NOT used in the TRWS subsolver.",(size_t)0,boolVec));
	addArgument(Size_TArgument<>(nesterovParameter_.maxNumberOfPresolveIterations(), "", "maxNumberOfPresolveIterations", "The number of TRWS iterations used a a presolver of the Nesterov algorithm",false));
	addArgument(DoubleArgument<>(nesterovParameter_.presolveMinRelativeDualImprovement(), "", "presolveMinRelativeDualImprovement", "The minimal improvement of the dual function in presolver. If the actual improvement is less, it stops the presolver",false));
	addArgument(Size_TArgument<>(nesterovParameter_.maxPrimalBoundIterationNumber(), "", "maxPrimalBoundIterationNumber", "The maximal iteration number of the transportation solver for estimating the primal fractal solution",false));
	addArgument(DoubleArgument<>(nesterovParameter_.primalBoundRelativePrecision(), "", "primalBoundRelativePrecision", "The relative precision used to solve the transportation problem for estimating the primal fractal solution",false));
	addArgument(BoolArgument(nesterovParameter_.verbose(), "", "debugverbose", "If set the solver will output debug information to the stdout"));
	addArgument(StringArgument<>(smoothingStrategyType, "", "smoothingStrategy", "Select smoothing strategy: ADAPTIVE_DIMINISHING, WC_DIMINISHING, ADAPTIVE_PRECISIONORIENTED or WC_PRECISIONORIENTED. Default is ADAPTIVE_DIMINISHING", false,stringVecSmoothStrategy));
	std::vector<std::string> stringgradientStepTypeVec(3); stringgradientStepTypeVec[0]="ADAPTIVE_STEP"; stringgradientStepTypeVec[1]="WC_STEP";  stringgradientStepTypeVec[2]="JOJIC_STEP";
	addArgument(StringArgument<>(gradientStepType, "", "gradientStep", "Select gradient step type: ADAPTIVE_STEP, WC_STEP, JOJIC_STEP. Default is ADAPTIVE_STEP", false,stringgradientStepTypeVec));
	addArgument(BoolArgument(nesterovParameter_.plainGradient_, "", "plainGradient", "If set the solver will use plain gradient updates instead of the accelerated gradient one"));
	addArgument(DoubleArgument<>(nesterovParameter_.gamma0_, "", "gamma0", "Internal parameter of the accelerated gradient algorithm. Do not change it without knowing why",false));
}

template <class IO, class GM, class ACC>
inline void NesterovCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running Nesterov caller" << std::endl;
   nesterovParameter_.isAbsolutePrecision()=(relativePrecision==0);
   nesterovParameter_.decompositionType()=trws_base::DecompositionStorage<GM>::getStructureType(stringDecompositionType);
   nesterovParameter_.setStartSmoothingValue(startSmoothingValue);
   nesterovParameter_.lazyLPPrimalBoundComputation()=(lazyLPPrimalBoundComputation==1);
   nesterovParameter_.setFastComputations((slowComputations==0));
   nesterovParameter_.setCanonicalNormalization((noNormalization==0));
   nesterovParameter_.setPrecision(precision);
   nesterovParameter_.smoothingStrategy()=NesterovType::Parameter::SmoothingParametersType::getSmoothingStrategyType(smoothingStrategyType);
   nesterovParameter_.gradientStep_=NesterovType::Parameter::getGradientStepType(gradientStepType);

   this-> template infer<NesterovType, TimingVisitorType, typename NesterovType::Parameter>(model, output, verbose, nesterovParameter_);
}

template <class IO, class GM, class ACC>
const std::string NesterovCaller<IO, GM, ACC>::name_ = "Nesterov";

} // namespace interface

} // namespace opengm

#endif /* NESTEROV_CALLER_HXX_ */
