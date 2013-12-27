/*
 * smooth_nesterov.hxx
 *
 *  Created on: Dec 23, 2013
 *      Author: bsavchyn
 */

#ifndef SMOOTH_NESTEROV_HXX_
#define SMOOTH_NESTEROV_HXX_

namespace opengm{
namespace gradient_base{
class DDvariable
{

};

}//namespace gradient_base

template<class GM>
class NesterovAcceleratedGradient
{
	template<class VISITOR>
	InferenceTermination infer(VISITOR & visitor)

private:
	ValueType _evaluateGradient(DDvariable& gradient);
	ValueType _evaluateSmoothObjective(DDvariable& point);
	DDvariable _currentDualVector;

	Storage 	  _storage;
	SumProdSolver _sumprodsolver;
	MaxSumSolver  _maxsumsolver;
};

template<class GM>
ValueType NesterovAcceleratedGradient<GM>::_evaluateGradient(DDvariable& gradient)
{
	//compute marginals
	std::for_each(_sumProdSolvers.begin(), _sumProdSolvers.end(), std::mem_fun(&SumProdSolver::Move));
	std::for_each(_sumProdSolvers.begin(), _sumProdSolvers.end(), std::mem_fun(&SumProdSolver::MoveBack));

	//transform marginals to dual vector
	gradientIt=gradient.begin();
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
ValueType NesterovAcceleratedGradient<GM>::_SetDualVariables(DDvariable& lambda)
{
	DDvariable lambda1=lambda-_currentDualVector;
	_currentDualVector=lambda;
	for (IndexType varId=0;varId<storage.masterModel().numberOfVariables();++varId)// all variables
	{ const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);

	  if (varList.size()==1) continue;
	  typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
	  IndexType firstModelId=modelIt->subModelId_;
	  IndexType firstModelVariableId=modelIt->subVariableId_;
	  ++modeIt;
	  for(;modelIt!=varList.end();++modelIt) //all related models
	  {
		  std::transform(lambda1.!!!begin(),lambda1.!!!end(),_storage.subModel(modelIt->subModelId_).ufBegin(modelIt->subVariableId_),
		  		  	     _storage.subModel(modelIt->subModelId_).ufBegin(modelIt->subVariableId_),std::plus<ValueType>());

		  std::transform(_storage.subModel(firstModelId).ufBegin(firstModelVariableId),
				         _storage.subModel(firstModelId).ufEnd(firstModelVariableId),
				          lambda1.!!!begin(),_storage.subModel(firstModelId).ufBegin(firstModelVariableId),
				          std::minus<ValueType>());
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
	   ValueType norm2=gradient.norm2();
	   do
	   {
		   omega*=2.0;
		   lambda=y+gradient/omega;
		   _SetDualVariables(lambda);
		   ValueType newObjVal=_evaluateSmoothObjective();
	   }
	   while ( newObjVal < (oldObjVal+norm2/2.0/omega));

	   //updating parameters
	   alpha=(sqrt(gamma+4*omega*gamma)-gamma)/omega/2.0;
	   gamma*=(1-alpha);
	   v+=alpha*gradient/gamma;
	   y=alpha*v+(1-alpha)*lambda;

	   //check stopping condition
	   //update smoothing
   }

}

}//namespace opengm

#endif /* SMOOTH_NESTEROV_HXX_ */
