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
	...
}

template<class GM>
ValueType NesterovAcceleratedGradient<GM>::_SetDualVariables(DDvariable& lambda)
{
	DDvariable lambda1=lambda-_currentDualVector;
	_currentDualVector=lambda;
	for (IndexType varId=0;varId<storage.masterModel().numberOfVariables();++varId)// all variables
	 for (IndexType modelId=0;modelId<storage.masterModel().numberOfModels();++modelId) //all models
	  for (LabelType label=0;label<storage.masterModel().numberOfLabels(varID);++label)//all labeles
	  {
		  std::transform(begin,end,_storage..ufBegin(_currentUnaryIndex),_storage.ufBegin(_currentUnaryIndex),std::plus<ValueType>());
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
