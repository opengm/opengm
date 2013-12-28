/*
 * smooth_nesterov.hxx
 *
 *  Created on: Dec 23, 2013
 *      Author: bsavchyn
 */

#ifndef SMOOTH_NESTEROV_HXX_
#define SMOOTH_NESTEROV_HXX_
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/auxiliary/primal_lpbound.hxx>


namespace opengm{

template<class GM>
struct Nesterov_Parameter : public PrimalLPBound_Parameter<typename GM::ValueType>
{
	typedef PrimalLPBound_Parameter<typename GM::ValueType> parent;
	typedef typename GM::ValueType ValueType;
	typedef trws_base::DecompositionStorage<GM> Storage;
	Nesterov_Parameter(
			size_t maxNumberOfIterations,
			ValueType precision,
			bool absolutePrecision=false,
			bool verbose=false,
			typename Storage::StructureType decompositionType=Storage::GENERALSTRUCTURE,
			bool fastComputations=true,
			ValueType gamma0=1.0,
			ValueType primalBoundRelativePrecision=std::numeric_limits<ValueType>::epsilon(),
			size_t maxPrimalBoundIterationNumber=100
			):
			parent(primalBoundRelativePrecision,maxPrimalBoundIterationNumber),
			maxNumberOfIterations_(maxNumberOfIterations),
			precision_(precision),
			absolutePrecision_(absolutePrecision),
			verbose_(verbose),
			decompositionType_(decompositionType),
			fastComputations_(fastComputations),
			gamma0_(gamma0){}

	size_t maxNumberOfIterations_;
	ValueType precision_;
	bool absolutePrecision_;
	bool verbose_;
	typename Storage::StructureType decompositionType_;
	bool fastComputations_;
	ValueType gamma0_;

#ifdef TRWS_DEBUG_OUTPUT
	  void print(std::ostream& fout)const
	  {
		  fout << "maxNumberOfIterations="<< maxNumberOfIterations_<<std::endl;
		  fout << "precision=" <<precision_<<std::endl;
		  fout <<"isAbsolutePrecision=" << absolutePrecision_<< std::endl;
		  fout <<"verbose="<<verbose_<<std::endl;

		  if (decompositionType_==Storage::GENERALSTRUCTURE)
			  fout <<"decompositionType=" <<"GENERAL"<<std::endl;
		  else if (decompositionType_==Storage::GRIDSTRUCTURE)
			  fout <<"decompositionType=" <<"GRID"<<std::endl;
		  else
			  fout <<"decompositionType=" <<"UNKNOWN"<<std::endl;

		  fout <<"fastComputations="<<fastComputations_<<std::endl;


		  /*
		   * Fractional primal bound estimator parameters
		   */
		  fout <<"maxPrimalBoundIterationNumber="<<parent::maxIterationNumber_<<std::endl;
		  fout <<"primalBoundRelativePrecision=" <<parent::relativePrecision_<<std::endl;
	  }
#endif
};

template<class GM, class ACC>
class NesterovAcceleratedGradient : public Inference<GM, ACC>
{
public:
	  typedef Inference<GM, ACC> parent;
	  typedef ACC AccumulationType;
	  typedef GM GraphicalModelType;
	  OPENGM_GM_TYPE_TYPEDEFS;

	typedef std::vector<typename GM::ValueType> DDvariable;
	typedef Nesterov_Parameter<GM> Parameter;
	typedef PrimalLPBound<GM,ACC> PrimalBoundEstimator;
	typedef trws_base::FunctionParameters<GM> FactorProperties;
	typedef trws_base::DecompositionStorage<GM> Storage;
	typedef typename trws_base::SumProdSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> SumProdSolverType;
	typedef typename trws_base::MaxSumSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> MaxSumSolverType;

	typedef visitors::ExplicitVerboseVisitor<NesterovAcceleratedGradient<GM, ACC> > VerboseVisitorType;
    typedef visitors::ExplicitTimingVisitor <NesterovAcceleratedGradient<GM, ACC> > TimingVisitorType;
	typedef visitors::ExplicitEmptyVisitor  <NesterovAcceleratedGradient<GM, ACC> > EmptyVisitorType;

	NesterovAcceleratedGradient(const GraphicalModelType& gm,const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			  ,std::ostream& fout=std::cout
#endif
			  )
	:
#ifdef TRWS_DEBUG_OUTPUT
	  _fout(param.verbose_ ? fout : *OUT::nullstream::Instance()),//_fout(fout),
#endif
	_parameters(param),
	_storage(gm,param.decompositionType_),
	_factorProperties(_storage.masterModel()),
	_estimator(gm,param),
	_bestPrimalLPbound(ACC::template neutral<ValueType>()),
	_bestPrimalBound(ACC::template neutral<ValueType>()),
	_bestDualBound(ACC::template ineutral<ValueType>()),
	_bestIntegerBound(ACC::template neutral<ValueType>()),
	_bestIntegerLabeling(_storage.masterModel().numberOfVariables(),0.0),
	_currentDualVector(_getDualVectorSize(_storage),0.0)
	{
		if (param.maxNumberOfIterations_==0) throw std::runtime_error("NesterovAcceleratedGradient: a strictly positive number of iterations must be provided!");

		_sumprodsolvers.resize(_storage.numberOfModels());
		_maxsumsolvers.resize(_storage.numberOfModels());
		for (size_t modelId=0;modelId<_sumprodsolvers.size();++modelId)
		{
			_sumprodsolvers[modelId]= new SumProdSolverType(_storage.subModel(modelId),_factorProperties,_parameters.fastComputations_);
			_maxsumsolvers[modelId]= new MaxSumSolverType(_storage.subModel(modelId),_factorProperties,_parameters.fastComputations_);
		}

	}

	~NesterovAcceleratedGradient()
	{
		std::for_each(_sumprodsolvers.begin(),_sumprodsolvers.end(),trws_base::DeallocatePointer<SumProdSolverType>);
		std::for_each(_maxsumsolvers.begin(),_maxsumsolvers.end(),trws_base::DeallocatePointer<MaxSumSolverType>);
	};

	template<class VISITOR>
	InferenceTermination infer(VISITOR & visitor);

	std::string name() const{ return "NEST"; }
	const GraphicalModelType& graphicalModel() const { return _storage.masterModel(); }
	InferenceTermination infer(){EmptyVisitorType visitor; return infer(visitor);};
	InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	{out = _bestIntegerLabeling;
	 return opengm::NORMAL;}

	ValueType bound() const{return _bestDualBound;}
	ValueType value() const{return _bestIntegerBound;}

private:
	ValueType _evaluateGradient(DDvariable* pgradient);
	ValueType _evaluateSmoothObjective(const DDvariable& point);
	ValueType _getSmoothObjective();
	size_t    _getDualVectorSize(const Storage& storage);
	void _SetDualVariables(const DDvariable& lambda);
	ValueType _estimateOmega0();

#ifdef TRWS_DEBUG_OUTPUT
	  std::ostream& _fout;
#endif
    Parameter 	  _parameters;
	Storage 	  _storage;
	FactorProperties _factorProperties;
	PrimalBoundEstimator 	_estimator;

	ValueType     _bestPrimalLPbound;
	ValueType     _bestPrimalBound;//best primal bound overall
	ValueType     _bestDualBound;
	ValueType     _bestIntegerBound;
	std::vector<LabelType> _bestIntegerLabeling;
	DDvariable 	  _currentDualVector;

	std::vector<SumProdSolverType*> _sumprodsolvers;
	std::vector<MaxSumSolverType*>  _maxsumsolvers;

};

template<class GM, class ACC>
typename NesterovAcceleratedGradient<GM,ACC>::ValueType
NesterovAcceleratedGradient<GM,ACC>::_evaluateGradient(DDvariable* pgradient)
{
	//compute marginals
	std::for_each(_sumprodsolvers.begin(), _sumprodsolvers.end(), std::mem_fun(&SumProdSolverType::Move));
	std::for_each(_sumprodsolvers.begin(), _sumprodsolvers.end(), std::mem_fun(&SumProdSolverType::MoveBack));

	//transform marginals to dual vector
	typename DDvariable::iterator gradientIt=pgradient->begin();
	for (IndexType varId=0;varId<_storage.masterModel().numberOfVariables();++varId)// all variables
	{ const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);

	  if (varList.size()==1) continue;
	  typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
	  IndexType firstModelId=modelIt->subModelId_;
	  IndexType firstModelVariableId=modelIt->subVariableId_;
	  typename SumProdSolverType::const_iterators_pair  firstMarginalsIt=_sumprodsolvers[firstModelId]->GetMarginals(firstModelVariableId);
	  ++modelIt;
	  for(;modelIt!=varList.end();++modelIt) //all related models
	  {
		  typename SumProdSolverType::const_iterators_pair  marginalsIt=_sumprodsolvers[modelIt->subModelId_]->GetMarginals(modelIt->subVariableId_);
		  gradientIt=std::transform(marginalsIt.first,marginalsIt.second,firstMarginalsIt.first,gradientIt,std::minus<ValueType>());
	  }
	}

	return _getSmoothObjective();
}

template<class GM, class ACC>
void NesterovAcceleratedGradient<GM,ACC>::_SetDualVariables(const DDvariable& lambda)
{
	DDvariable delta=lambda-_currentDualVector;
	_currentDualVector=lambda;
	typename DDvariable::const_iterator deltaIt=delta.begin();
	for (IndexType varId=0;varId<_storage.masterModel().numberOfVariables();++varId)// all variables
	{ const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);

	  if (varList.size()==1) continue;
	  typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
	  IndexType firstModelId=modelIt->subModelId_;
	  IndexType firstModelVariableId=modelIt->subVariableId_;
	  ++modelIt;
	  for(;modelIt!=varList.end();++modelIt) //all related models
	  {
		  std::transform(_storage.subModel(modelIt->subModelId_).ufBegin(modelIt->subVariableId_),
				         _storage.subModel(modelIt->subModelId_).ufEnd(modelIt->subVariableId_),
				          deltaIt,_storage.subModel(modelIt->subModelId_).ufBegin(modelIt->subVariableId_),
				          std::plus<ValueType>());

		  std::transform(_storage.subModel(firstModelId).ufBegin(firstModelVariableId),
				         _storage.subModel(firstModelId).ufEnd(firstModelVariableId),
				          deltaIt,_storage.subModel(firstModelId).ufBegin(firstModelVariableId),
				          std::minus<ValueType>());
		  deltaIt+=_storage.masterModel().numberOfLabels(varId);
	  }
	}
}

template<class GM, class ACC>
template<class VISITOR>
InferenceTermination NesterovAcceleratedGradient<GM,ACC>::infer(VISITOR & visitor)
{
	//_ComputeInitialSmoothing();//TODO: look to ADSal - fixed smoothing

   DDvariable gradient(_currentDualVector.size()),
		      lambda(_currentDualVector.size()),
		      y(_currentDualVector),
		      v(_currentDualVector);
   ValueType   alpha,
   	   	   	   gamma=_parameters.gamma0_,
		       omega=_estimateOmega0();


   omega=omega/2.0;

   for (size_t i=0;i<_parameters.maxNumberOfIterations_;++i)
   {
	   //gradient step with approximate linear search:
	   ValueType oldObjVal=_evaluateGradient(&gradient);
	   ValueType norm2=std::inner_product(gradient.begin(),gradient.end(),gradient.begin(),(ValueType)0);//squared L2 norm
	   ValueType newObjVal;
	   do
	   {
		   omega*=2.0;
		   //lambda=y+gradient/omega;
		   std::transform(gradient.begin(),gradient.end(),lambda.begin(),std::bind1st(std::multiplies<ValueType>(),1/omega));
		   std::transform(y.begin(),y.end(),lambda.begin(),lambda.begin(),std::plus<ValueType>());

		   newObjVal=_evaluateSmoothObjective(lambda);
	   }
	   while ( newObjVal < (oldObjVal+norm2/2.0/omega));

	   //updating parameters
	   alpha=(sqrt(gamma+4*omega*gamma)-gamma)/omega/2.0;
	   gamma*=(1-alpha);
	   //v+=(alpha/gamma)*gradient;
	   trws_base::transform_inplace(gradient.begin(),gradient.end(),std::bind1st(std::multiplies<ValueType>(),alpha/gamma));
	   std::transform(v.begin(),v.end(),gradient.begin(),gradient.begin(),std::plus<ValueType>());

	   //y=alpha*v+(1-alpha)*lambda;
	   trws_base::transform_inplace(lambda.begin(),lambda.end(),std::bind1st(std::multiplies<ValueType>(),(1-alpha)));
	   trws_base::transform_inplace(v.begin(),v.end(),std::bind1st(std::multiplies<ValueType>(),alpha));
	   std::transform(v.begin(),v.end(),lambda.begin(),v.begin(),std::plus<ValueType>());

	   //check stopping condition
	   //update smoothing
   }

}

}//namespace opengm

#endif /* SMOOTH_NESTEROV_HXX_ */
