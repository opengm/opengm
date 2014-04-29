#ifndef TRWS_ADSAL_HXX_
#define TRWS_ADSAL_HXX_
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/auxiliary/primal_lpbound.hxx>
#include <opengm/inference/trws/smoothing_strategy.hxx>

namespace opengm{

template<class ValueType,class GM>
struct ADSal_Parameter : public trws_base::SmoothingBasedInference_Parameter<ValueType,GM>

{
	typedef trws_base::DecompositionStorage<GM> Storage;
	typedef typename trws_base::SmoothingBasedInference_Parameter<ValueType,GM> parent;
	typedef typename parent::SmoothingParametersType SmoothingParametersType;
	typedef typename parent::SumProdSolverParametersType SumProdSolverParametersType;
	typedef typename parent::MaxSumSolverParametersType MaxSumSolverParametersType;
	typedef typename parent::PrimalLPEstimatorParametersType PrimalLPEstimatorParametersType;
	typedef typename parent::SmoothingStrategyType SmoothingStrategyType;

	ADSal_Parameter(size_t numOfExternalIterations=0,
			    ValueType precision=1.0,
			    bool absolutePrecision=true,
			    size_t numOfInternalIterations=3,
			    typename Storage::StructureType decompositionType=Storage::GENERALSTRUCTURE,
			    ValueType smoothingGapRatio=4,
			    ValueType startSmoothingValue=0.0,
			    ValueType primalBoundPrecision=std::numeric_limits<ValueType>::epsilon(),
			    size_t maxPrimalBoundIterationNumber=100,
			    size_t presolveMaxIterNumber=100,
			    bool canonicalNormalization=true,
			    ValueType presolveMinRelativeDualImprovement=0.01,
			    bool lazyLPPrimalBoundComputation=true,
			    bool lazyDerivativeComputation=true,
			    ValueType smoothingDecayMultiplier=-1.0,
			    //bool worstCaseSmoothing=false,
			    SmoothingStrategyType smoothingStrategy=SmoothingParametersType::ADAPTIVE_DIMINISHING,
			    bool fastComputations=true,
			    bool verbose=false
			    )
	 :parent(numOfExternalIterations,
			 precision,
			 absolutePrecision,
			 numOfInternalIterations,
			 decompositionType,
			 smoothingGapRatio,
			 startSmoothingValue,
			 primalBoundPrecision,
			 maxPrimalBoundIterationNumber,
			 presolveMaxIterNumber,
			 canonicalNormalization,
			 presolveMinRelativeDualImprovement,
			 lazyLPPrimalBoundComputation,
			 smoothingDecayMultiplier,
			 //worstCaseSmoothing,
			 smoothingStrategy,
			 fastComputations,
			 verbose
			 ),
	  lazyDerivativeComputation_(lazyDerivativeComputation)
	  {};

	  bool lazyDerivativeComputation_;

	  bool& lazyDerivativeComputation(){return lazyDerivativeComputation_;}
	  const bool& lazyDerivativeComputation()const {return lazyDerivativeComputation_;}

#ifdef TRWS_DEBUG_OUTPUT
	  void print(std::ostream& fout)const
	  {
		  parent::print(fout);
		  fout <<"lazyDerivativeComputation="<< lazyDerivativeComputation()<< std::endl;
	  }
#endif
};

//! [class adsal]
/// ADSal - adaptive diminishing smoothing algorithm
/// Based on the paper:
/// B. Savchynskyy, S. Schmidt, J. H. Kappes, C. Schnörr
/// Efficient MRF Energy Minimization via Adaptive Diminishing Smoothing, In UAI, 2012, pp. 746-755
///
/// it provides:
/// * primal integer approximate solution for MRF energy minimization problem
/// * approximate primal and dual solutions of the local polytope relaxation of the problem.
/// Duality gap converges to zero in the limit and can be used as an accuracy measure of the algorithm.
///
///
/// TODO: Code can be significantly speeded up!
///
/// Corresponding author: Bogdan Savchynskyy
///
///\ingroup inference

template<class GM, class ACC>
class ADSal : public trws_base::SmoothingBasedInference<GM, ACC> //public Inference<GM, ACC>,SmoothingBasedInference<GM, ACC>
{
public:
	  typedef trws_base::SmoothingBasedInference<GM, ACC> parent;
	  typedef ACC AccumulationType;
	  typedef GM GraphicalModelType;
	  OPENGM_GM_TYPE_TYPEDEFS;

	  typedef typename parent::Storage Storage;
	  typedef typename parent::SumProdSolver SumProdSolver;
	  typedef typename parent::MaxSumSolver MaxSumSolver;
	  typedef typename parent::PrimalBoundEstimator PrimalBoundEstimator;

	  typedef ADSal_Parameter<ValueType,GM> Parameter;

	  //typedef visitors::ExplicitVerboseVisitor<ADSal<GM, ACC> > VerboseVisitorType;
	  typedef visitors::VerboseVisitor<ADSal<GM, ACC> > VerboseVisitorType;
	  //typedef visitors::ExplicitTimingVisitor <ADSal<GM, ACC> > TimingVisitorType;//TODO: fix it
	  typedef visitors::TimingVisitor <ADSal<GM, ACC> > TimingVisitorType;
	  //typedef visitors::ExplicitEmptyVisitor  <ADSal<GM, ACC> > EmptyVisitorType;
	  typedef visitors::EmptyVisitor  <ADSal<GM, ACC> > EmptyVisitorType;

	  ADSal(const GraphicalModelType& gm,const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			  ,std::ostream& fout=std::cout
#endif
			  )
	  :
	   parent(gm,param
#ifdef TRWS_DEBUG_OUTPUT
			  ,(param.verbose_ ? fout : *OUT::nullstream::Instance())
#endif
			   ),
	   _parameters(param)
	  {
#ifdef TRWS_DEBUG_OUTPUT
	  parent::_fout << "Parameters of the "<< name() <<" algorithm:"<<std::endl;
	  param.print(parent::_fout);
#endif

		  if (param.numOfExternalIterations_==0) throw std::runtime_error("ADSal: a strictly positive number of iterations must be provided!");
	  };

	  std::string name() const{ return "ADSal"; }
	  InferenceTermination infer(){EmptyVisitorType visitor; return infer(visitor);};
	  template<class VISITOR> InferenceTermination infer(VISITOR & visitor);

	  /*
	   * for testing only!
	   */
	  InferenceTermination oldinfer();
private:
	  Parameter _parameters;
};

template<class GM,class ACC>
template<class VISITOR>
InferenceTermination ADSal<GM,ACC>::infer(VISITOR & vis)
{
	trws_base::VisitorWrapper<VISITOR,ADSal<GM, ACC> > visitor(&vis,this);
	size_t counter=0;//!> oracle calls counter

	visitor.addLog("oracleCalls");
	visitor.addLog("primalLPbound");
	visitor.begin();

	if (parent::_sumprodsolver.GetSmoothing()<=0.0)
	{
		if (parent::_Presolve(visitor, &counter)==CONVERGENCE)
		{
			parent::_SelectOptimalBoundsAndLabeling();
			visitor();
			visitor.log("oracleCalls",(double)counter);
			visitor.log("primalLPbound",(double)parent::_bestPrimalLPbound);

			visitor.end();
			return NORMAL;
		}
#ifdef TRWS_DEBUG_OUTPUT
		parent::_fout <<"Switching to the smooth solver============================================"<<std::endl;
#endif
		counter+=parent::_EstimateStartingSmoothing(visitor);
	}

	bool forwardMoveNeeded=true;
   for (size_t i=0;i<_parameters.numOfExternalIterations_;++i)
   {
#ifdef TRWS_DEBUG_OUTPUT
	   parent::_fout <<"Main iteration Nr."<<i<<"============================================"<<std::endl;
#endif

	   InferenceTermination returncode;
	   counter+=_parameters.numberOfInternalIterations();
	   if (forwardMoveNeeded)
	   {
		++counter;returncode=parent::_sumprodsolver.infer();
	    forwardMoveNeeded=false;
	   }
	   else
		returncode=parent::_sumprodsolver.core_infer();

	   if (returncode==CONVERGENCE)
	   {
		   parent::_SelectOptimalBoundsAndLabeling();
		   visitor();
		   visitor.log("oracleCalls",(double)counter);
		   visitor.log("primalLPbound",(double)parent::_bestPrimalLPbound);
		   visitor.end();
		   return NORMAL;
	   }


	   ++counter;parent::_maxsumsolver.ForwardMove();//initializes a move, makes a forward move and computes the dual bound, is used also in derivative computation in the next line
#ifdef TRWS_DEBUG_OUTPUT
	   parent::_fout << "_maxsumsolver.bound()=" <<parent::_maxsumsolver.bound()<<std::endl;
#endif

	   ValueType derivative;
	   if (parent::_isLPBoundComputed() || !_parameters.lazyDerivativeComputation())
	   {
		   ++counter;  parent::_sumprodsolver.GetMarginalsAndDerivativeMove();
	    derivative=parent::_EstimateRhoDerivative();
#ifdef TRWS_DEBUG_OUTPUT
	    parent::_fout << "derivative="<<derivative<<std::endl;
#endif
	    forwardMoveNeeded=true;
	   }
	   else
		   derivative=parent::_FastEstimateRhoDerivative();

	   if ( parent::_CheckStoppingCondition(&returncode))
	   {
		   visitor();
		   visitor.log("oracleCalls",(double)counter);
		   visitor.log("primalLPbound",(double)parent::_bestPrimalLPbound);
		   visitor.end();
		   return NORMAL;
	   }


		size_t flag=visitor();
		visitor.log("oracleCalls",(double)counter);
		visitor.log("primalLPbound",(double)parent::_bestPrimalLPbound);
		if( flag != visitors::VisitorReturnFlag::ContinueInf ){
			break;
		}

	   if (parent::_UpdateSmoothing(parent::_bestPrimalBound,parent::_maxsumsolver.bound(),parent::_sumprodsolver.bound(),derivative,i+1))
	   	  forwardMoveNeeded=true;
   }

   parent::_SelectOptimalBoundsAndLabeling();
   visitor();
   visitor.log("oracleCalls",(double)counter);
   visitor.log("primalLPbound",(double)parent::_bestPrimalLPbound);
   visitor.end();

	return NORMAL;
}

template<class GM,class ACC>
InferenceTermination ADSal<GM,ACC>::oldinfer()
{
	if (parent::_sumprodsolver.GetSmoothing()<=0.0)
	{
		parent::_EstimateStartingSmoothing();
	}

   for (size_t i=0;i<_parameters.numOfExternalIterations_;++i)
   {
#ifdef TRWS_DEBUG_OUTPUT
	   parent::_fout <<"Main iteration Nr."<<i<<"============================================"<<std::endl;
#endif
	   for (size_t innerIt=0;innerIt<_parameters.maxNumberOfIterations_;++innerIt)
	   {
		   parent::_sumprodsolver.ForwardMove();
		   parent::_sumprodsolver.BackwardMove();
#ifdef TRWS_DEBUG_OUTPUT
	    parent::_fout <<"subIter="<< innerIt<<", smoothDualBound=" << parent::_sumprodsolver.bound() <<std::endl;
#endif
	   }

	   parent::_sumprodsolver.ForwardMove();
	   parent::_sumprodsolver.GetMarginalsAndDerivativeMove();
	   parent::_maxsumsolver.ForwardMove();//initializes a move, makes a forward move and computes the dual bound, is used also in derivative computation in the next line
	   ValueType derivative=parent::_EstimateRhoDerivative();
#ifdef TRWS_DEBUG_OUTPUT
	   parent::_fout << "derivative="<<derivative<<std::endl;
#endif
	   InferenceTermination returncode;
	   if ( parent::_CheckStoppingCondition(&returncode)) return returncode;

	   parent::_UpdateSmoothing(parent::_bestPrimalBound,parent::_maxsumsolver.bound(),parent::_sumprodsolver.bound(),derivative,i+1);
   }
	return opengm::NORMAL;
}

}
#endif
