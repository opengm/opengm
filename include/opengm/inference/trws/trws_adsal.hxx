#ifndef TRWS_ADSAL_HXX_
#define TRWS_ADSAL_HXX_
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/auxiliary/primal_lpbound.hxx>

namespace opengm{

template<class ValueType,class GM>
struct ADSal_Parameter : public trws_base::SumProdTRWS_Parameters<ValueType>, public PrimalLPBound_Parameter<ValueType>
{
	typedef trws_base::DecompositionStorage<GM> Storage;
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
			    bool worstCaseSmoothing=false,
			    bool verbose=false
			    )
	 :trws_base::SumProdTRWS_Parameters<ValueType>(numOfInternalIterations,startSmoothingValue,precision,absolutePrecision,2*std::numeric_limits<ValueType>::epsilon(),true,canonicalNormalization),
	  PrimalLPBound_Parameter<ValueType>(primalBoundPrecision,maxPrimalBoundIterationNumber),
	  numOfExternalIterations_(numOfExternalIterations),
	  presolveMaxIterNumber_(presolveMaxIterNumber),
	  decompositionType_(decompositionType),
	  smoothingGapRatio_(smoothingGapRatio),
	  presolveMinRelativeDualImprovement_(presolveMinRelativeDualImprovement),
	  canonicalNormalization_(canonicalNormalization),
	  lazyLPPrimalBoundComputation_(lazyLPPrimalBoundComputation),
	  lazyDerivativeComputation_(lazyDerivativeComputation),
	  smoothingDecayMultiplier_(smoothingDecayMultiplier),
	  worstCaseSmoothing_(worstCaseSmoothing),
	  verbose_(verbose)
	  {};

	  size_t numOfExternalIterations_;
	  size_t presolveMaxIterNumber_;
	  typename Storage::StructureType decompositionType_;
	  ValueType smoothingGapRatio_;
	  ValueType presolveMinRelativeDualImprovement_;
	  bool canonicalNormalization_;
	  bool lazyLPPrimalBoundComputation_;
	  bool lazyDerivativeComputation_;
	  ValueType smoothingDecayMultiplier_;//!> forces smoothing decay as smoothingDecayMultiplier_/iterationCounter
	  bool worstCaseSmoothing_;
	  bool verbose_;

	  /*
	   * Main algorithm parameters
	   */
	  size_t& maxNumberOfIterations(){return numOfExternalIterations_;}
	  const size_t& maxNumberOfIterations()const {return numOfExternalIterations_;}

	  size_t& numberOfInternalIterations(){return trws_base::SumProdTRWS_Parameters<ValueType>::maxNumberOfIterations_;}
	  const size_t& numberOfInternalIterations()const{return trws_base::SumProdTRWS_Parameters<ValueType>::maxNumberOfIterations_;}

	  ValueType& precision(){return trws_base::SumProdTRWS_Parameters<ValueType>::precision_;}
	  const ValueType& precision()const{return trws_base::SumProdTRWS_Parameters<ValueType>::precision_;}

	  bool&      isAbsolutePrecision(){return trws_base::SumProdTRWS_Parameters<ValueType>::absolutePrecision_;}
	  const bool&      isAbsolutePrecision()const {return trws_base::SumProdTRWS_Parameters<ValueType>::absolutePrecision_;}

	  ValueType& smoothingGapRatio(){return smoothingGapRatio_;}
	  const ValueType& smoothingGapRatio()const{return smoothingGapRatio_;}

	  bool& lazyLPPrimalBoundComputation(){return lazyLPPrimalBoundComputation_;}
	  const bool& lazyLPPrimalBoundComputation()const{return lazyLPPrimalBoundComputation_;}

	  bool& lazyDerivativeComputation(){return lazyDerivativeComputation_;}
	  const bool& lazyDerivativeComputation()const {return lazyDerivativeComputation_;}

	  ValueType& smoothingDecayMultiplier(){return smoothingDecayMultiplier_;}
	  const ValueType& smoothingDecayMultiplier()const{return smoothingDecayMultiplier_;}

	  bool& worstCaseSmoothing(){return worstCaseSmoothing_;}
	  const bool& worstCaseSmoothing()const{return worstCaseSmoothing_;}

	  typename Storage::StructureType& decompositionType(){return decompositionType_;}
	  const typename Storage::StructureType& decompositionType()const{return decompositionType_;}

	  ValueType& startSmoothingValue(){return trws_base::SumProdTRWS_Parameters<ValueType>::smoothingValue_;}
	  const ValueType& startSmoothingValue()const{return trws_base::SumProdTRWS_Parameters<ValueType>::smoothingValue_;}

	  bool& fastComputations(){return trws_base::SumProdTRWS_Parameters<ValueType>::fastComputations_;}
	  const bool& fastComputations()const{return trws_base::SumProdTRWS_Parameters<ValueType>::fastComputations_;}

	  bool& canonicalNormalization(){return canonicalNormalization_;}
	  const bool& canonicalNormalization()const{return canonicalNormalization_;}

	  /*
	   * Presolve parameters
	   */
	  size_t& maxNumberOfPresolveIterations(){return presolveMaxIterNumber_;}
	  const size_t& maxNumberOfPresolveIterations()const{return presolveMaxIterNumber_;}

	  ValueType& presolveMinRelativeDualImprovement() {return presolveMinRelativeDualImprovement_;}
	  const ValueType& presolveMinRelativeDualImprovement()const {return presolveMinRelativeDualImprovement_;}
	  /*
	   * Fractional primal bound estimator parameters
	   */
	  size_t& maxPrimalBoundIterationNumber(){return PrimalLPBound_Parameter<ValueType>::maxIterationNumber_;}
	  const size_t& maxPrimalBoundIterationNumber()const{return PrimalLPBound_Parameter<ValueType>::maxIterationNumber_;}

	  ValueType& primalBoundRelativePrecision(){return PrimalLPBound_Parameter<ValueType>::relativePrecision_;}
	  const ValueType& primalBoundRelativePrecision()const{return PrimalLPBound_Parameter<ValueType>::relativePrecision_;}

	  bool& verbose(){return verbose_;}
	  const bool& verbose()const{return verbose_;}

#ifdef TRWS_DEBUG_OUTPUT
	  void print(std::ostream& fout)const
	  {
		  fout << "maxNumberOfIterations="<< maxNumberOfIterations()<<std::endl;
		  fout << "numberOfInternalIterations="<< numberOfInternalIterations()<<std::endl;
		  fout << "precision=" <<precision()<<std::endl;
		  fout <<"isAbsolutePrecision=" << isAbsolutePrecision()<< std::endl;
		  fout <<"smoothingGapRatio="  << smoothingGapRatio()<< std::endl;
		  fout <<"lazyLPPrimalBoundComputation="<<lazyLPPrimalBoundComputation()<< std::endl;
		  fout <<"lazyDerivativeComputation="<< lazyDerivativeComputation()<< std::endl;
		  fout <<"smoothingDecayMultiplier=" << smoothingDecayMultiplier()<< std::endl;
		  fout <<"worstCaseSmoothing="<<worstCaseSmoothing()<<std::endl;

		  if (decompositionType()==Storage::GENERALSTRUCTURE)
			  fout <<"decompositionType=" <<"GENERAL"<<std::endl;
		  else if (decompositionType()==Storage::GRIDSTRUCTURE)
			  fout <<"decompositionType=" <<"GRID"<<std::endl;
		  else
			  fout <<"decompositionType=" <<"UNKNOWN"<<std::endl;

		  fout <<"startSmoothingValue=" << startSmoothingValue()<<std::endl;
		  fout <<"fastComputations="<<fastComputations()<<std::endl;
		  fout <<"canonicalNormalization="<<canonicalNormalization()<<std::endl;

		  /*
		   * Presolve parameters
		   */
		  fout <<"maxNumberOfPresolveIterations="<<maxNumberOfPresolveIterations()<<std::endl;
		  fout <<"presolveMinRelativeDualImprovement=" <<presolveMinRelativeDualImprovement()<<std::endl;

		  /*
		   * Fractional primal bound estimator parameters
		   */
		  fout <<"maxPrimalBoundIterationNumber="<<maxPrimalBoundIterationNumber()<<std::endl;
		  fout <<"primalBoundRelativePrecision=" <<primalBoundRelativePrecision()<<std::endl;
		  fout <<"verbose="<<verbose()<<std::endl;
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
/// Duality gap comverges to zero in the limit andcan be used as an accuracy measure of the algorithm.
///
///
/// TODO: Code can be significantly speeded up!
///
/// Corresponding author: Bogdan Savchynskyy
///
///\ingroup inference

template<class GM, class ACC>
class ADSal : public Inference<GM, ACC>
{
public:
	  typedef Inference<GM, ACC> parent;
	  typedef ACC AccumulationType;
	  typedef GM GraphicalModelType;
	  OPENGM_GM_TYPE_TYPEDEFS;

	  typedef trws_base::DecompositionStorage<GM> Storage;
	  typedef trws_base::SumProdTRWS<GM,ACC> SumProdSolver;
	  typedef trws_base::MaxSumTRWS<GM,ACC> MaxSumSolver;
	  typedef PrimalLPBound<GM,ACC> PrimalBoundEstimator;

	  typedef ADSal_Parameter<ValueType,GM> Parameter;

	  typedef VerboseVisitor<ADSal<GM, ACC> > VerboseVisitorType;
	  typedef TimingVisitor <ADSal<GM, ACC> > TimingVisitorType;
	  typedef EmptyVisitor  <ADSal<GM, ACC> > EmptyVisitorType;

	  typedef typename MaxSumSolver::ReparametrizerType ReparametrizerType;

	  ADSal(const GraphicalModelType& gm,const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			  ,std::ostream& fout=std::cout
#endif
			  )
	  :_parameters(param),
	  _storage(gm,param.decompositionType_),
	  _sumprodsolver(_storage,param
#ifdef TRWS_DEBUG_OUTPUT
			  ,(param.verbose_ ? fout : *OUT::nullstream::Instance()) //fout
#endif
			  ),
	  _maxsumsolver(_storage,typename MaxSumSolver::Parameters(param.presolveMaxIterNumber_,param.precision_,param.absolutePrecision_,param.presolveMinRelativeDualImprovement_,param.fastComputations_,param.canonicalNormalization_)
#ifdef TRWS_DEBUG_OUTPUT
			  ,(param.verbose_ ? fout : *OUT::nullstream::Instance())//fout
#endif
			  ),
	  _estimator(gm,param),
#ifdef TRWS_DEBUG_OUTPUT
	  _fout(param.verbose_ ? fout : *OUT::nullstream::Instance()),//_fout(fout),
#endif
	  _bestPrimalLPbound(ACC::template neutral<ValueType>()),
	  _bestPrimalBound(ACC::template neutral<ValueType>()),
	  _bestDualBound(ACC::template ineutral<ValueType>()),
	  _bestIntegerBound(ACC::template neutral<ValueType>()),
	  _bestIntegerLabeling(_storage.masterModel().numberOfVariables(),0.0),
	  _initializationStage(true)
	  {
#ifdef TRWS_DEBUG_OUTPUT
	  _fout << "Parameters of the "<< name() <<" algorithm:"<<std::endl;
	  param.print(_fout);
#endif

		  if (param.numOfExternalIterations_==0) throw std::runtime_error("ADSal: a strictly positive number of iterations must be provided!");
	  };

	  std::string name() const{ return "ADSal"; }
	  const GraphicalModelType& graphicalModel() const { return _storage.masterModel(); }
	  InferenceTermination infer(){EmptyVisitorType visitor; return infer(visitor);};
	  template<class VISITOR> InferenceTermination infer(VISITOR & visitor);
	  InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	  {out = _bestIntegerLabeling;
	   return opengm::NORMAL;}

	  ValueType bound() const{return _bestDualBound;}
	  ValueType value() const{return _bestIntegerBound;}

	  void getTreeAgreement(std::vector<bool>& out,std::vector<LabelType>* plabeling=0,std::vector<std::vector<LabelType> >* ptreeLabelings=0){_maxsumsolver.getTreeAgreement(out,plabeling,ptreeLabelings);}
	  Storage& getDecompositionStorage(){return _storage;}
	  const typename MaxSumSolver::FactorProperties& getFactorProperties()const {return _maxsumsolver.getFactorProperties();}
	  ReparametrizerType* getReparametrizer(const typename ReparametrizerType::Parameter& params=typename ReparametrizerType::Parameter())const
	  {return _maxsumsolver.getReparametrizer(params);}

	  /*
	   * for testing only!
	   */
	  InferenceTermination oldinfer();
private:
	  template<class VISITOR>
	  InferenceTermination _Presolve(VISITOR& visitor);
	  template<class VISITOR>
	  void _EstimateStartingSmoothing(VISITOR& visitor);
	  /*
	   * use iterationCounterPlus1=0 if you run _UpdateSmoothing for estmation of the initial smoothing
	   */
	  bool _UpdateSmoothing(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound, ValueType derivativeValue,size_t iterationCounterPlus1=0);
	  bool _CheckStoppingCondition(InferenceTermination*);
	  void _UpdatePrimalEstimator();
	  ValueType _EstimateRhoDerivative()const;
	  ValueType _FastEstimateRhoDerivative()const{return (_sumprodsolver.bound()-_maxsumsolver.bound())/_sumprodsolver.GetSmoothing();}
	  ValueType _ComputeStartingWorstCaseSmoothing(ValueType primalBound,ValueType dualBound)const;
	  ValueType _ComputeWorstCaseSmoothing(ValueType primalBound,ValueType smoothDualBound)const;
	  ValueType _ComputeSmoothingMultiplier()const;
	  LabelType _ComputeMaxNumberOfLabels()const;
	  bool _SmoothingMustBeDecreased(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound,std::pair<ValueType,ValueType>* lhsRhs)const;
	  bool _isLPBoundComputed()const;
	  void _SelectOptimalBoundsAndLabeling();

	  Parameter _parameters;
	  Storage 				_storage;
	  SumProdSolver			_sumprodsolver;
	  MaxSumSolver          _maxsumsolver;
	  PrimalBoundEstimator 	_estimator;
#ifdef TRWS_DEBUG_OUTPUT
	  std::ostream& _fout;
#endif
	  ValueType     _bestPrimalLPbound;
	  ValueType     _bestPrimalBound;//best primal bound overall

	  ValueType     _bestDualBound;
	  ValueType     _bestIntegerBound;
	  std::vector<LabelType> _bestIntegerLabeling;

	  bool _initializationStage;//!>used to inform smoothing selection functions, that we are in the smoothing intialization stage
	  /*
	   * optimization of computations
	   */
	  typename SumProdSolver::OutputContainerType _marginalsTemp;
};


template<class GM,class ACC>
void ADSal<GM,ACC>::_SelectOptimalBoundsAndLabeling()
{
	//Best integer bound...
	if (ACC::bop(_sumprodsolver.value(),_maxsumsolver.value()))
		{
		_bestIntegerLabeling=_sumprodsolver.arg();
		 _bestIntegerBound=_sumprodsolver.value();
		}else
		{
		 _bestIntegerLabeling=_maxsumsolver.arg();
		 _bestIntegerBound=_maxsumsolver.value();
		}

	//Best primalBound
	ACC::op(_bestPrimalLPbound,_bestIntegerBound,_bestPrimalBound);
#ifdef TRWS_DEBUG_OUTPUT
	_fout << "_bestPrimalBound=" <<_bestPrimalBound<<std::endl;
#endif

	//Best dual bound...
	if (ACC::ibop(_sumprodsolver.bound(),_maxsumsolver.bound()))
		 _bestDualBound=_sumprodsolver.bound();
	else
		 _bestDualBound=_maxsumsolver.bound();

}

template<class GM,class ACC>
template<class VISITOR>
void ADSal<GM,ACC>::_EstimateStartingSmoothing(VISITOR& visitor)
{
	_sumprodsolver.SetSmoothing(_ComputeStartingWorstCaseSmoothing(_maxsumsolver.value(),_maxsumsolver.bound()));
#ifdef TRWS_DEBUG_OUTPUT
	_fout <<"_maxsumsolver.value()="<<_maxsumsolver.value()<<", _maxsumsolver.bound()="<<_maxsumsolver.bound()<<std::endl;
	_fout << "WorstCaseSmoothing="<<_ComputeStartingWorstCaseSmoothing(_maxsumsolver.value(),_maxsumsolver.bound())<<std::endl;
#endif
	std::pair<ValueType,ValueType> lhsRhs;
	_sumprodsolver.ForwardMove();
	_sumprodsolver.GetMarginalsAndDerivativeMove();

	if (!_parameters.worstCaseSmoothing_)
	{
	visitor(_maxsumsolver.value(),_maxsumsolver.bound());
	do{
	ValueType derivative=_EstimateRhoDerivative();
	_parameters.smoothingGapRatio_*=2;//!> increase the ratio to obtain fulfillment of a smoothing changing condition
	_UpdateSmoothing(_maxsumsolver.value(),_maxsumsolver.bound(),_sumprodsolver.bound(),derivative);
	_parameters.smoothingGapRatio_/=2;//!> restoring the normal value before checking the condition
	_sumprodsolver.ForwardMove();
	_sumprodsolver.GetMarginalsAndDerivativeMove();
	visitor(_maxsumsolver.value(),_maxsumsolver.bound());
	}while (_SmoothingMustBeDecreased(_maxsumsolver.value(),_maxsumsolver.bound(),_sumprodsolver.bound(),&lhsRhs));
	}else
		_UpdateSmoothing(_maxsumsolver.value(),_maxsumsolver.bound(),_sumprodsolver.bound(),_EstimateRhoDerivative());
}


template<class GM,class ACC>
template<class VISITOR>
opengm::InferenceTermination ADSal<GM,ACC>::_Presolve(VISITOR& visitor)
{
#ifdef TRWS_DEBUG_OUTPUT
	 _fout << "Running TRWS presolve..."<<std::endl;
#endif
	 return _maxsumsolver.infer_visitor_updates(visitor);
}

template<class GM,class ACC>
typename ADSal<GM,ACC>::LabelType ADSal<GM,ACC>::_ComputeMaxNumberOfLabels()const
{
	LabelType numOfLabels=0;
	for (size_t i=0;i<_storage.numberOfSharedVariables();++i)
		numOfLabels=std::max(numOfLabels,_storage.numberOfLabels(i));

	return numOfLabels;
}

template<class GM,class ACC>
typename ADSal<GM,ACC>::ValueType ADSal<GM,ACC>::_ComputeSmoothingMultiplier()const
{
	ValueType multiplier=0;
	ValueType logLabels=log((ValueType)_ComputeMaxNumberOfLabels());
	for (size_t i=0;i<_storage.numberOfModels();++i)
		multiplier+=_storage.size(i)*logLabels;

	return multiplier;
}

template<class GM,class ACC>
typename ADSal<GM,ACC>::ValueType ADSal<GM,ACC>::_ComputeStartingWorstCaseSmoothing(ValueType primalBound,ValueType dualBound)const
{
	return fabs((primalBound-dualBound)/_ComputeSmoothingMultiplier()/(2.0*_parameters.smoothingGapRatio_-1));
}

template<class GM,class ACC>
typename ADSal<GM,ACC>::ValueType ADSal<GM,ACC>::_ComputeWorstCaseSmoothing(ValueType primalBound,ValueType smoothDualBound)const
{
	return fabs((primalBound-smoothDualBound)/_ComputeSmoothingMultiplier()/(2.0*_parameters.smoothingGapRatio_));
}

template<class GM,class ACC>
bool ADSal<GM,ACC>::_SmoothingMustBeDecreased(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound,std::pair<ValueType,ValueType>* lhsRhs)const
{
	if (!_parameters.worstCaseSmoothing_)
	{
	lhsRhs->first=dualBound-smoothDualBound;
	lhsRhs->second=(primalBound-smoothDualBound)/(2*_parameters.smoothingGapRatio_);
	if (_parameters.smoothingDecayMultiplier_ <= 0.0 || _initializationStage)
		return ACC::ibop(lhsRhs->first,lhsRhs->second);
	else
		return true;
	}else if (_ComputeWorstCaseSmoothing(primalBound,smoothDualBound)<_sumprodsolver.GetSmoothing())
		return true;

	return false;
}

template<class GM,class ACC>
bool ADSal<GM,ACC>::_UpdateSmoothing(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound, ValueType derivativeValue,size_t iterationCounterPlus1)
{
#ifdef TRWS_DEBUG_OUTPUT
	_fout << "dualBound="<<dualBound<<", smoothDualBound="<<smoothDualBound<<", derivativeValue="<<derivativeValue<<std::endl;
#endif

	std::pair<ValueType,ValueType> lhsRhs;
	if (_SmoothingMustBeDecreased(primalBound,dualBound,smoothDualBound,&lhsRhs) || _initializationStage)
	{
	ValueType newsmoothing;

	if (!_parameters.worstCaseSmoothing_)
	  newsmoothing=_sumprodsolver.GetSmoothing() - (lhsRhs.second - lhsRhs.first)*(2.0*_parameters.smoothingGapRatio_)/(2.0*_parameters.smoothingGapRatio_-1.0)/derivativeValue;
	else
	  newsmoothing=_ComputeWorstCaseSmoothing(primalBound,smoothDualBound);

	if ( (_parameters.smoothingDecayMultiplier_ > 0.0) && (!_initializationStage) )
	{
	 ValueType newMulIt=_parameters.smoothingDecayMultiplier_*iterationCounterPlus1+1;
	 ValueType oldMulIt=_parameters.smoothingDecayMultiplier_*(iterationCounterPlus1-1)+1;
	 newsmoothing=std::min(newsmoothing,_sumprodsolver.GetSmoothing()*oldMulIt/newMulIt);
	}

	if (newsmoothing > 0)
	if ((newsmoothing < _sumprodsolver.GetSmoothing()) ||  _initializationStage) _sumprodsolver.SetSmoothing(newsmoothing);
#ifdef TRWS_DEBUG_OUTPUT
	 _fout << "smoothing changed to " <<  _sumprodsolver.GetSmoothing()<<std::endl;
#endif
	 return true;
	}
	return false;
}

template<class GM,class ACC>
void ADSal<GM,ACC>::_UpdatePrimalEstimator()
{
 std::pair<ValueType,ValueType> bestNorms=std::make_pair((ValueType)0.0,(ValueType)0.0);
 ValueType numberOfVariables=_storage.masterModel().numberOfVariables();
 for (IndexType var=0;var<numberOfVariables;++var)
 {
	 _marginalsTemp.resize(_storage.numberOfLabels(var));
	 std::pair<ValueType,ValueType> norms=_sumprodsolver.GetMarginals(var, _marginalsTemp.begin());

	 bestNorms.second=std::max(bestNorms.second,norms.second);
	 bestNorms.first+=norms.first*norms.first;

	 transform_inplace(_marginalsTemp.begin(),_marginalsTemp.end(),trws_base::make0ifless<ValueType>(norms.second));//!> remove what is less than the precision

	 TransportSolver::_Normalize(_marginalsTemp.begin(),_marginalsTemp.end(),(ValueType)0.0);
	 _estimator.setVariable(var,_marginalsTemp.begin());
 }
#ifdef TRWS_DEBUG_OUTPUT
 _fout << "l2 gradient norm="<<sqrt(bestNorms.first)<<", "<<"l_inf gradient norm="<<bestNorms.second<<std::endl;
#endif
}

template<class GM,class ACC>
bool ADSal<GM,ACC>::_isLPBoundComputed()const
{
	return (!_parameters.lazyLPPrimalBoundComputation_ || !ACC::bop(_sumprodsolver.value(),_bestPrimalBound) );
}

template<class GM,class ACC>
bool ADSal<GM,ACC>::_CheckStoppingCondition(InferenceTermination* preturncode)
{
  if( _isLPBoundComputed())
  {
    _UpdatePrimalEstimator();

	ACC::op(_estimator.getTotalValue(),_bestPrimalLPbound);
#ifdef TRWS_DEBUG_OUTPUT
	_fout << "_primalLPbound=" <<_estimator.getTotalValue()<<std::endl;
#endif
  }
    _SelectOptimalBoundsAndLabeling();

	if (_maxsumsolver.CheckTreeAgreement(preturncode)) return true;

	if (_sumprodsolver.CheckDualityGap(_bestPrimalBound,_maxsumsolver.bound()))
	{
#ifdef TRWS_DEBUG_OUTPUT
	  _fout << "ADSal::_CheckStoppingCondition(): Precision attained! Problem solved!"<<std::endl;
#endif
	 *preturncode=CONVERGENCE;
	 return true;
	}

	return false;
}


template<class GM,class ACC>
template<class VISITOR>
InferenceTermination ADSal<GM,ACC>::infer(VISITOR & vis)
{
	trws_base::VisitorWrapper<VISITOR,ADSal<GM, ACC> > visitor(&vis,this);

	visitor.begin(value(),bound());

	if (_sumprodsolver.GetSmoothing()<=0.0)
	{
		if (_Presolve(visitor)==CONVERGENCE)
		{
			_SelectOptimalBoundsAndLabeling();
			visitor.end(value(), bound());
			return NORMAL;
		}
#ifdef TRWS_DEBUG_OUTPUT
		_fout <<"Switching to the smooth solver============================================"<<std::endl;
#endif
		_EstimateStartingSmoothing(visitor);
	}

	_initializationStage=false;

	bool forwardMoveNeeded=true;
   for (size_t i=0;i<_parameters.numOfExternalIterations_;++i)
   {
#ifdef TRWS_DEBUG_OUTPUT
	   _fout <<"Main iteration Nr."<<i<<"============================================"<<std::endl;
#endif

	   InferenceTermination returncode;
	   if (forwardMoveNeeded)
	   {
	    returncode=_sumprodsolver.infer();
	    forwardMoveNeeded=false;
	   }
	   else
		returncode=_sumprodsolver.core_infer();

	   if (returncode==CONVERGENCE)
	   {
		   _SelectOptimalBoundsAndLabeling();
		   visitor.end(value(), bound());
		   return NORMAL;
//		   return returncode;
	   }

	   _maxsumsolver.ForwardMove();//initializes a move, makes a forward move and computes the dual bound, is used also in derivative computation in the next line
#ifdef TRWS_DEBUG_OUTPUT
	   _fout << "_maxsumsolver.bound()=" <<_maxsumsolver.bound()<<std::endl;
#endif

	   ValueType derivative;
	   if (_isLPBoundComputed() || !_parameters.lazyDerivativeComputation())
	   {
	    _sumprodsolver.GetMarginalsAndDerivativeMove();
	    derivative=_EstimateRhoDerivative();
#ifdef TRWS_DEBUG_OUTPUT
	    _fout << "derivative="<<derivative<<std::endl;
#endif
	    forwardMoveNeeded=true;
	   }
	   else
		   derivative=_FastEstimateRhoDerivative();

	   if ( _CheckStoppingCondition(&returncode))
	   {
		   visitor.end(value(), bound());
		   return NORMAL;
//		   return returncode;
	   }

	   visitor(value(),bound());
	   if (_UpdateSmoothing(_bestPrimalBound,_maxsumsolver.bound(),_sumprodsolver.bound(),derivative,i+1))
		   forwardMoveNeeded=true;
   }

   _SelectOptimalBoundsAndLabeling();
   visitor.end(value(), bound());

	return NORMAL;
}

template<class GM,class ACC>
InferenceTermination ADSal<GM,ACC>::oldinfer()
{
	if (_sumprodsolver.GetSmoothing()<=0.0)
	{
		_EstimateStartingSmoothing();
		//if (_Presolve()==NORMAL) return NORMAL;
	}

   for (size_t i=0;i<_parameters.numOfExternalIterations_;++i)
   {
#ifdef TRWS_DEBUG_OUTPUT
	   _fout <<"Main iteration Nr."<<i<<"============================================"<<std::endl;
#endif
	   for (size_t innerIt=0;innerIt<_parameters.maxNumberOfIterations_;++innerIt)
	   {
	    _sumprodsolver.ForwardMove();
	    _sumprodsolver.BackwardMove();
#ifdef TRWS_DEBUG_OUTPUT
	    _fout <<"subIter="<< innerIt<<", smoothDualBound=" << _sumprodsolver.bound() <<std::endl;
#endif
	   }

	   _sumprodsolver.ForwardMove();
	   _sumprodsolver.GetMarginalsAndDerivativeMove();
	   _maxsumsolver.ForwardMove();//initializes a move, makes a forward move and computes the dual bound, is used also in derivative computation in the next line
	   ValueType derivative=_EstimateRhoDerivative();
#ifdef TRWS_DEBUG_OUTPUT
	   _fout << "derivative="<<derivative<<std::endl;
#endif
	   InferenceTermination returncode;
	   if ( _CheckStoppingCondition(&returncode)) return returncode;

	   _UpdateSmoothing(_bestPrimalBound,_maxsumsolver.bound(),_sumprodsolver.bound(),derivative,i+1);
   }
	return opengm::NORMAL;
}

template<class GM,class ACC>
typename ADSal<GM,ACC>::ValueType
ADSal<GM,ACC>::_EstimateRhoDerivative()const
{
	ValueType derivative=0.0;
	for (size_t i=0;i<_storage.numberOfModels();++i)
	{
		ValueType delta;
		ACC::op(_sumprodsolver.getDerivative(i),(_sumprodsolver.getBound(i)-_maxsumsolver.getBound(i))/_sumprodsolver.GetSmoothing(),delta);
		derivative+=delta;
	}
	return derivative;
}

}
#endif
