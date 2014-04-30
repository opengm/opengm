/*
 * combilp_caller.hxx
 *
 *  Created on: May 22, 2013
 *      Author: bsavchyn
 */

#ifndef COMBILP_CALLER_HXX_
#define COMBILP_CALLER_HXX_

#include <opengm/opengm.hxx>
#include <opengm/inference/trws/trws_trws.hxx>
#include <opengm/inference/trws/trws_adsal.hxx>
#include <opengm/inference/combilp.hxx>

#include "inference_caller_base.hxx"
#include "../argument/argument.hxx"

namespace opengm {

namespace interface {

template <class IO, class GM, class ACC>
class CombiLPCaller : public InferenceCallerBase<IO, GM, ACC, CombiLPCaller<IO, GM, ACC> > {
protected:
   typedef InferenceCallerBase<IO, GM, ACC, CombiLPCaller<IO, GM, ACC> > BaseClass;
   using BaseClass::addArgument;
   using BaseClass::io_;
   using BaseClass::infer;
   typedef typename BaseClass::OutputBase OutputBase;

   typename TRWSi<GM,ACC>::Parameter trwsParameter_;
   typename ADSal<GM,ACC>::Parameter adsalParameter_;

   virtual void runImpl(GM& model, OutputBase& output, const bool verbose);

   std::string lpsolvertype;
   bool parameter_verbose;
   std::string parameter_reparametrizedFileName;
   bool parameter_saveProblemMasks;
   std::string parameter_maskFileNamePre;
   size_t parameter_maxNumberOfILPCycles;

   size_t LPSolver_maxNumberOfIterations;
   double LPSolver_parameter_precision;
   size_t LPSolver_relativePrecision;
   std::string LPSolver_stringDecompositionType;
   size_t LPSolver_slowComputations;
   size_t LPSolver_noNormalization;

   size_t adsalParameter_lazyLPPrimalBoundComputation;
   size_t adsalParameter_lazyDerivativeComputation;
   double adsalParameter_startSmoothingValue;

   size_t trwsParameter_treeAgreeMaxStableIter;
public:
   const static std::string name_;
   CombiLPCaller(IO& ioIn);
};

template <class IO, class GM, class ACC>
inline CombiLPCaller<IO, GM, ACC>::CombiLPCaller(IO& ioIn)
   : BaseClass(name_, "detailed description of the internal CombiLP caller...", ioIn),
     adsalParameter_startSmoothingValue(adsalParameter_.startSmoothingValue())
     {
	std::vector<size_t> boolVec(2); boolVec[0]=0; boolVec[1]=1;
	std::vector<std::string> stringVec(3); stringVec[0]="GENERAL"; stringVec[1]="GRID"; stringVec[2]="EDGE";
	std::vector<std::string> lpsolver(2); lpsolver[0]="TRWSi"; lpsolver[1]="ADSal";
	addArgument(StringArgument<>(lpsolvertype, "", "lpsolve", "Select local polytope solver : TRWSi or ADSal. Default is TRWSi", lpsolver.front(), lpsolver));
	addArgument(BoolArgument(parameter_verbose, "", "debugverbose", "If set the solver will output debug information to the stdout"));
	addArgument(StringArgument<>(parameter_reparametrizedFileName, "", "savedfilename", "If set to a valid filename the reparametrized model will be saved", std::string("")));
	addArgument(BoolArgument(parameter_saveProblemMasks, "", "saveProblemMasks", "Saves masks of the subproblems passed to the ILP solver"));
	addArgument(StringArgument<>(parameter_maskFileNamePre, "", "maskFileNamePre", "Path and filename prefix of the subproblem masks, see parameter saveProblemMasks", std::string("")));
	//addArgument(Size_TArgument<>(parameter_maxNumberOfILPCycles, "", "maxNumberOfILPCycles", "Max number of ILP solver cycles",false));
	addArgument(Size_TArgument<>(parameter_maxNumberOfILPCycles, "", "maxNumberOfILPCycles", "Max number of ILP solver cycles",(size_t)100));


	//LP solver parameters:
	addArgument(Size_TArgument<>(LPSolver_maxNumberOfIterations, "", "maxIt", "Maximum number of iterations.",true));
	addArgument(DoubleArgument<>(LPSolver_parameter_precision, "", "precision", "Duality gap based absolute precision to be obtained. Default is 0.0. Use parameter --relative to select the relative one",(double)0.0));
	addArgument(Size_TArgument<>(LPSolver_relativePrecision, "", "relative", "If set to 1 , then the parameter --precision determines a relative precision value. Default is an absolute one",(size_t)0,boolVec));
	addArgument(StringArgument<>(LPSolver_stringDecompositionType, "d", "decomposition", "Select decomposition: GENERAL, GRID or EDGE. Default is GENERAL", false,stringVec));
	addArgument(Size_TArgument<>(LPSolver_slowComputations, "", "slowComputations", "If set to 1 the type of the pairwise factors (Potts for now, will be extended to truncated linear and quadratic) will NOT be used to speed up computations of the TRWS subsolver.",(size_t)0,boolVec));
	addArgument(Size_TArgument<>(LPSolver_noNormalization, "", "noNormalization", "If set to 1 the canonical normalization is NOT used in the TRWS subsolver.",(size_t)0,boolVec));

	// specific TRWSi parameters
	addArgument(DoubleArgument<>(trwsParameter_.minRelativeDualImprovement(), "", "TRWS_minRelativeDualImprovement", "TRWSi::The minimal improvement of the dual function. If the actual improvement is less, it stops the solver",false));
	//addArgument(Size_TArgument<>(trwsParameter_.treeAgreeMaxStableIter_, "", "treeAgreeMaxStableIter", "Maximum number of iterations after the last improvements of the tree agreement.",false));
	addArgument(Size_TArgument<>(trwsParameter_treeAgreeMaxStableIter, "", "treeAgreeMaxStableIter", "Maximum number of iterations after the last improvements of the tree agreement.",(size_t)0));

	// specific ADSal parameters
	addArgument(Size_TArgument<>(adsalParameter_.numberOfInternalIterations(), "", "ADSal_numberOfInternalIterations", "ADSal::Number of internal iterations (between changes of smoothing).",false));
	addArgument(DoubleArgument<>(adsalParameter_.smoothingGapRatio(), "", "ADSal_smoothingGapRatio", "ADSal::Constant gamma - smoothing gap ratio",false));
	addArgument(Size_TArgument<>(adsalParameter_lazyLPPrimalBoundComputation, "", "ADSal_lazyLPPrimalBoundComputation", "ADSal::If set to 1 the fractal primal bound is not computed when the primal bound improved over the last outer iteration",(size_t)0,boolVec));
    addArgument(Size_TArgument<>(adsalParameter_lazyDerivativeComputation, "", "ADSal_lazyDerivativeComputation", "ADSal::If set to 1 the derivative w.r.t. smoothing is computed only when the fractal primal bound computation is needed",(size_t)0,boolVec));
	addArgument(DoubleArgument<>(adsalParameter_.smoothingDecayMultiplier(), "", "ADSal_smoothingDecayMultiplier", "ADSal::Smoothing decay parameter defines the rate of a forced smoothing decay. Default is <= 0, which switches off the forced decay",false));
//	addArgument(DoubleArgument<>(adsalParameter_.startSmoothingValue(), "", "ADSal_startSmoothing", "ADSal::Starting smoothing value. Default is automatic selection",false));
	addArgument(DoubleArgument<>(adsalParameter_startSmoothingValue, "", "ADSal_startSmoothing", "ADSal::Starting smoothing value. Default is automatic selection",false));
	addArgument(Size_TArgument<>(adsalParameter_.maxNumberOfPresolveIterations(), "", "ADSal_maxNumberOfPresolveIterations", "ADSal::The number of TRWS iterations used a a presolver of the ADSal algorithm",false));
	addArgument(DoubleArgument<>(adsalParameter_.presolveMinRelativeDualImprovement(), "", "ADSal_presolveMinRelativeDualImprovement", "ADSal::The minimal improvement of the dual function in presolver. If the actual improvement is less, it stops the presolver",false));
	addArgument(Size_TArgument<>(adsalParameter_.maxPrimalBoundIterationNumber(), "", "ADSal_maxPrimalBoundIterationNumber", "ADSal::The maximal iteration number of the transportation solver for estimating the primal fractal solution",false));
	addArgument(DoubleArgument<>(adsalParameter_.primalBoundRelativePrecision(), "", "ADSal_primalBoundRelativePrecision", "ADSal::The relative precision used to solve the transportation problem for estimating the primal fractal solution",false));
}

template <class IO, class GM, class ACC>
inline void CombiLPCaller<IO, GM, ACC>::runImpl(GM& model, OutputBase& output, const bool verbose) {
   std::cout << "running internal CombiLP caller" << std::endl;

   if (lpsolvertype=="TRWSi")
   {
	   typedef TRWSi<GM,ACC> LPSOLVER;

	   typedef CombiLP<GM, ACC, LPSOLVER> CombiLPType;
	   typedef typename CombiLPType::VerboseVisitorType VerboseVisitorType;
	   typedef typename CombiLPType::EmptyVisitorType EmptyVisitorType;
	   typedef typename CombiLPType::TimingVisitorType TimingVisitorType;
	   typename CombiLPType::Parameter parameter_;
	   typename LPSOLVER::Parameter lpsolverParameter_(trwsParameter_);

	   lpsolverParameter_.setTreeAgreeMaxStableIter(trwsParameter_treeAgreeMaxStableIter);
	   lpsolverParameter_.maxNumberOfIterations_=LPSolver_maxNumberOfIterations;
	   lpsolverParameter_.precision()=LPSolver_parameter_precision;
	   lpsolverParameter_.isAbsolutePrecision()=(LPSolver_relativePrecision==0);
//	   lpsolverParameter_.decompositionType()=(LPSolver_stringDecompositionType.compare("GRID")==0 ? trws_base::DecompositionStorage<GM>::GRIDSTRUCTURE :
//			   trws_base::DecompositionStorage<GM>::GENERALSTRUCTURE );
	   lpsolverParameter_.decompositionType()=trws_base::DecompositionStorage<GM>::getStructureType(LPSolver_stringDecompositionType);
	   lpsolverParameter_.fastComputations()=(LPSolver_slowComputations==0);
	   lpsolverParameter_.canonicalNormalization()=(LPSolver_noNormalization==0);

	   parameter_.verbose_=lpsolverParameter_.verbose_=parameter_verbose;
	   parameter_.reparametrizedModelFileName_=parameter_reparametrizedFileName;
	   parameter_.saveProblemMasks_=parameter_saveProblemMasks;
	   parameter_.maskFileNamePre_=parameter_maskFileNamePre;
	   parameter_.maxNumberOfILPCycles_=parameter_maxNumberOfILPCycles;
	   parameter_.lpsolverParameter_=lpsolverParameter_;
	   this-> template infer<CombiLPType, TimingVisitorType, typename CombiLPType::Parameter>(model, output, verbose, parameter_);
   }else if (lpsolvertype=="ADSal")
   {
	   typedef ADSal<GM,ACC> LPSOLVER;

	   typedef CombiLP<GM, ACC, LPSOLVER> CombiLPType;
	   typedef typename CombiLPType::VerboseVisitorType VerboseVisitorType;
	   typedef typename CombiLPType::EmptyVisitorType EmptyVisitorType;
	   typedef typename CombiLPType::TimingVisitorType TimingVisitorType;
	   typename CombiLPType::Parameter parameter_;

	   adsalParameter_.lazyLPPrimalBoundComputation()=(adsalParameter_lazyLPPrimalBoundComputation==1);
	   adsalParameter_.lazyDerivativeComputation()=(adsalParameter_lazyLPPrimalBoundComputation==1);
	   adsalParameter_.setStartSmoothingValue(adsalParameter_startSmoothingValue);

	   typename LPSOLVER::Parameter lpsolverParameter_(adsalParameter_);

	   lpsolverParameter_.maxNumberOfIterations()=LPSolver_maxNumberOfIterations;
	   lpsolverParameter_.setPrecision(LPSolver_parameter_precision);
	   lpsolverParameter_.isAbsolutePrecision()=(LPSolver_relativePrecision==0);
	   lpsolverParameter_.decompositionType()=(LPSolver_stringDecompositionType.compare("GRID")==0 ? trws_base::DecompositionStorage<GM>::GRIDSTRUCTURE :
			   trws_base::DecompositionStorage<GM>::GENERALSTRUCTURE );
	   lpsolverParameter_.setFastComputations((LPSolver_slowComputations==0));
	   lpsolverParameter_.setCanonicalNormalization((LPSolver_noNormalization==0));

	   parameter_.verbose_=lpsolverParameter_.verbose_=parameter_verbose;
	   parameter_.reparametrizedModelFileName_=parameter_reparametrizedFileName;
	   parameter_.saveProblemMasks_=parameter_saveProblemMasks;
	   parameter_.maskFileNamePre_=parameter_maskFileNamePre;
	   parameter_.maxNumberOfILPCycles_=parameter_maxNumberOfILPCycles;
	   parameter_.lpsolverParameter_=lpsolverParameter_;
	   this-> template infer<CombiLPType, TimingVisitorType, typename CombiLPType::Parameter>(model, output, verbose, parameter_);
   }else throw RuntimeError("Unknown local polytope solver!");
}

template <class IO, class GM, class ACC>
const std::string CombiLPCaller<IO, GM, ACC>::name_ = "CombiLP";

} // namespace interface

} // namespace opengm

#endif /* COMBILP_CALLER_HXX_ */
