/*
 * smooth_nesterov.hxx
 *
 *  Created on: Dec 23, 2013
 *      Author: bsavchyn
 */

#ifndef SMOOTH_NESTEROV_HXX_
#define SMOOTH_NESTEROV_HXX_

namespace opengm{

template<class GM>
struct Nesterov_Parameter : public PrimalLPBound_Parameter<ValueType>
{
	typedef typename GM::ValueType ValueType;
	typedef trws_base::DecompositionStorage<GM> Storage;
	Nesterov_Parameter(
			size_t maxNumberOfIterations,
			ValueType precision,
			bool absolutePrecision=false,
			bool verbose=false,
			typename Storage::StructureType decompositionType=Storage::GENERALSTRUCTUR,
			bool fastComputations=true):
			maxNumberOfIterations_(maxNumberOfIterations),
			precision_(precision),
			absolutePrecision_(absolutePrecision),
			verbose_(verbose),
			decompositionType_(decompositionType),
			fastComputations_(fastComputations){}

	size_t maxNumberOfIterations_;
	ValueType precision_;
	bool absolutePrecision_;
	bool verbose_;
	typename Storage::StructureType decompositionType_;
	bool fastComputations_;
};

template<class GM>
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

	NesterovAcceleratedGradient(const GraphicalModelType& gm,const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			  ,std::ostream& fout=std::cout
#endif
			  )
	:_parameters(param),
	_storage(gm,param.decompositionType_),
	_estimator(gm,param)
	{//TODO: the constructor is incomplete!

	}

	template<class VISITOR>
	InferenceTermination infer(VISITOR & visitor)


private:
	ValueType _evaluateGradient(DDvariable& gradient);
	ValueType _evaluateSmoothObjective(DDvariable& point);

#ifdef TRWS_DEBUG_OUTPUT
	  std::ostream& _fout;
#endif

	Storage 	  _storage;
	std::vector<SumProdSolver*> _sumprodsolvers;
	std::vector<MaxSumSolver*>  _maxsumsolvers;
	PrimalBoundEstimator 	_estimator;

	DDvariable _currentDualVector;
};

template<class GM>
ValueType NesterovAcceleratedGradient<GM>::_evaluateGradient(DDvariable& gradient)
{
	//compute marginals
	std::for_each(_sumProdSolvers.begin(), _sumProdSolvers.end(), std::mem_fun(&SumProdSolver::Move));
	std::for_each(_sumProdSolvers.begin(), _sumProdSolvers.end(), std::mem_fun(&SumProdSolver::MoveBack));

	//transform marginals to dual vector
	DDvariable::iterator gradientIt=gradient.begin();
	for (IndexType varId=0;varId<storage.masterModel().numberOfVariables();++varId)// all variables
	{ const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);

	  if (varList.size()==1) continue;
	  typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
	  IndexType firstModelId=modelIt->subModelId_;
	  IndexType firstModelVariableId=modelIt->subVariableId_;
	  typename SumProdSolver::const_iterators_pair  fistrMarginalsIt=_sumProdSolver[firstModelId].GetMarginals(firstModelVariableId);
	  ++modeIt;
	  for(;modelIt!=varList.end();++modelIt) //all related models
	  {
		  typename SumProdSolver::const_iterators_pair  marginalsIt=_sumProdSolver[modelIt->subModelId_].GetMarginals(modelIt->subVariableId_);
		  gradientIt=std::transform(marginalsIt.first,marginalsIt.second,fistrMarginalsIt.first,gradientIt,std::minus<ValueType>());
	  }
	}
}

template<class GM>
ValueType NesterovAcceleratedGradient<GM>::_SetDualVariables(const DDvariable& lambda)
{
	DDvariable delta=lambda-_currentDualVector;
	_currentDualVector=lambda;
	DDvariable::const_iterator deltaIt=delta.begin();
	for (IndexType varId=0;varId<storage.masterModel().numberOfVariables();++varId)// all variables
	{ const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);

	  if (varList.size()==1) continue;
	  typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
	  IndexType firstModelId=modelIt->subModelId_;
	  IndexType firstModelVariableId=modelIt->subVariableId_;
	  ++modeIt;
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
		  deltaIt+=storage.masterModel().numberOfLabels(varId);
	  }
	}
}

template<class GM>
template<class VISITOR>
InferenceTermination NesterovAcceleratedGradient<GM>::infer(VISITOR & visitor)
{
	_ComputeInitialSmoothing();//TODO: look to ADSal - fixed smoothing

   DDvariable gradient, lamda,v,alpha;
   DDVariable y=v=_currentDualVector,
		      gamma=_parameters.gamma0_,
		      omega=_paramegters.omega0_;

   omega=omega/2.0;

   for (size_t i=0;i<_parameters.numOfExternalIterations_;++i)
   {
	   //gradient step with approximate linear search:
	   ValueType oldObjVal=_evaluateGradient(&gradient);
	   ValueType norm2=std::inner_product(gradient.begin(),gradient.end(),gradient.begin(),(ValueType)0);//squared L2 norm
	   do
	   {
		   omega*=2.0;
		   //lambda=y+gradient/omega;
		   std::transform(lambda.begin(),lambda.end(),lambda.begin(),std::bind1st(std::multiplies<ValueType>(),1/omega))
		   std::transform(y.begin(),y.end(),lambda.begin(),lambda.begin(),std::plus<ValueType>());

		   _SetDualVariables(lambda);
		   ValueType newObjVal=_evaluateSmoothObjective();
	   }
	   while ( newObjVal < (oldObjVal+norm2/2.0/omega));

	   //updating parameters
	   alpha=(sqrt(gamma+4*omega*gamma)-gamma)/omega/2.0;
	   gamma*=(1-alpha);
	   //v+=(alpha/gamma)*gradient;
	   transform_inplace(gradient.begin(),gradient.end(),std::bind1st(std::multiplies<ValueType>(),alpha/gamma));
	   std::transform(v.begin(),v.end(),gradient.begin(),gradient.begin(),std::plus<ValueType>());

	   //y=alpha*v+(1-alpha)*lambda;
	   transform_inplace(lambda.begin(),lambda.end(),std::bind1st(std::multiplies<ValueType>(),(1-alpha));
	   transform_inplace(v.begin(),v.end(),std::bind1st(std::multiplies<ValueType>(),alpha);
	   std::transform(v.begin(),v.end(),lambda.begin(),v.begin(),std::plus<ValueType>());

	   //check stopping condition
	   //update smoothing
   }

}

}//namespace opengm

#endif /* SMOOTH_NESTEROV_HXX_ */
