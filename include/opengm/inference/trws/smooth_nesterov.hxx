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
#include <opengm/inference/trws/smoothing_strategy.hxx>

namespace opengm{

template<class ValueType,class GM>
struct Nesterov_Parameter : public trws_base::SmoothingBasedInference_Parameter<ValueType,GM>

{
	typedef trws_base::DecompositionStorage<GM> Storage;
	typedef typename trws_base::SmoothingBasedInference_Parameter<ValueType,GM> parent;
	typedef typename parent::SmoothingParametersType SmoothingParametersType;
	typedef typename parent::SumProdSolverParametersType SumProdSolverParametersType;
	typedef typename parent::MaxSumSolverParametersType MaxSumSolverParametersType;
	typedef typename parent::PrimalLPEstimatorParametersType PrimalLPEstimatorParametersType;
	typedef typename parent::SmoothingStrategyType SmoothingStrategyType;

	Nesterov_Parameter(size_t numOfExternalIterations=0,
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
			 )
	  {};
};


template<class GM, class ACC>
class NesterovAcceleratedGradient : public trws_base::SmoothingBasedInference<GM, ACC> //Inference<GM, ACC>
{
public:
	typedef trws_base::SmoothingBasedInference<GM, ACC> parent;
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;

	//typedef std::vector<typename GM::ValueType> DDVectorType;

	typedef typename parent::Storage Storage;
	typedef typename Storage::DDVectorType DDVectorType;
	typedef typename parent::SumProdSolver SumProdSolver;
	typedef typename parent::MaxSumSolver MaxSumSolver;
	typedef typename parent::PrimalBoundEstimator PrimalBoundEstimator;

	typedef Nesterov_Parameter<ValueType,GM> Parameter;

	typedef visitors::ExplicitVerboseVisitor<NesterovAcceleratedGradient<GM, ACC> > VerboseVisitorType;
	typedef visitors::ExplicitTimingVisitor <NesterovAcceleratedGradient<GM, ACC> > TimingVisitorType;
	typedef visitors::ExplicitEmptyVisitor  <NesterovAcceleratedGradient<GM, ACC> > EmptyVisitorType;

	NesterovAcceleratedGradient(const GraphicalModelType& gm,const Parameter& param
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
		_parameters(param),
		_currentDualVector(_getDualVectorSize(),0.0)
	{
#ifdef TRWS_DEBUG_OUTPUT
		parent::_fout << "Parameters of the "<< name() <<" algorithm:"<<std::endl;
		param.print(parent::_fout);
#endif

		if (param.numOfExternalIterations_==0) throw std::runtime_error("NEST: a strictly positive number of iterations must be provided!");
	};

	template<class VISITOR>
	InferenceTermination infer(VISITOR & visitor);

	std::string name() const{ return "NEST"; }
	InferenceTermination infer(){EmptyVisitorType visitor; return infer(visitor);};


private:
	ValueType _evaluateGradient(const DDVectorType& point,DDVectorType* pgradient);
	ValueType _evaluateSmoothObjective(const DDVectorType& point,bool smoothingDerivativeEstimationNeeded=false);
	size_t    _getDualVectorSize()const{return parent::_storage.getDDVectorSize();}
	void      _SetDualVariables(const DDVectorType& lambda);
	ValueType _estimateOmega0()const{return 1;};//TODO: exchange with a reasonable value
	void _InitSmoothing();//TODO: refactor me

	ValueType getLipschitzConstant()const;

	Parameter 	  _parameters;
	DDVectorType 	  _currentDualVector;
};

template<class GM, class ACC>
typename NesterovAcceleratedGradient<GM,ACC>::ValueType
NesterovAcceleratedGradient<GM,ACC>::getLipschitzConstant()const
{
 ValueType result=0;
 for (IndexType modelId=0;modelId<parent::_storage.numberOfModels();++modelId)
	 result+=(ValueType)parent::_storage.size(modelId);

 return result/parent::_sumprodsolver.GetSmoothing();
}

template<class GM, class ACC>
typename NesterovAcceleratedGradient<GM,ACC>::ValueType
NesterovAcceleratedGradient<GM,ACC>::_evaluateGradient(const DDVectorType& point,DDVectorType* pgradient)
{
	ValueType bound=_evaluateSmoothObjective(point);
	std::vector<ValueType> buffer1st;
	std::vector<ValueType> buffer;
	//transform marginals to dual vector
	pgradient->resize(_currentDualVector.size());
	typename DDVectorType::iterator gradientIt=pgradient->begin();
	for (IndexType varId=0;varId<parent::_storage.masterModel().numberOfVariables();++varId)// all variables
	{
		const typename Storage::SubVariableListType& varList=parent::_storage.getSubVariableList(varId);

		if (varList.size()==1) continue;
		typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();
		IndexType firstModelId=modelIt->subModelId_;
		IndexType firstModelVariableId=modelIt->subVariableId_;
		buffer1st.resize(parent::_storage.masterModel().numberOfLabels(varId));
		buffer.resize(parent::_storage.masterModel().numberOfLabels(varId));
		parent::_sumprodsolver.GetMarginalsForSubModel(firstModelId,firstModelVariableId,buffer1st.begin());
		++modelIt;
		for(;modelIt!=varList.end();++modelIt) //all related models
		{
			parent::_sumprodsolver.GetMarginalsForSubModel(modelIt->subModelId_,modelIt->subVariableId_,buffer.begin());
			gradientIt=std::transform(buffer.begin(),buffer.end(),buffer1st.begin(),gradientIt,std::minus<ValueType>());
		}
	}

	return bound;
}


template<class GM, class ACC>
typename NesterovAcceleratedGradient<GM,ACC>::ValueType
NesterovAcceleratedGradient<GM,ACC>::_evaluateSmoothObjective(const DDVectorType& point,bool smoothingDerivativeEstimationNeeded)
{
	_SetDualVariables(point);
	parent::_sumprodsolver.ForwardMove();
	if (smoothingDerivativeEstimationNeeded)
	{
		parent::_sumprodsolver.GetMarginalsAndDerivativeMove();
	}else parent::_sumprodsolver.GetMarginalsMove();

	return parent::_sumprodsolver.bound();
}

template<class GM, class ACC>
void NesterovAcceleratedGradient<GM,ACC>::_SetDualVariables(const DDVectorType& lambda)
{
	DDVectorType delta(_currentDualVector.size());
	std::transform(lambda.begin(),lambda.end(),_currentDualVector.begin(),delta.begin(),std::minus<ValueType>());
	_currentDualVector=lambda;
	parent::_storage.addDDvector(delta);
};

template<class GM, class ACC>
void NesterovAcceleratedGradient<GM,ACC>::_InitSmoothing()
{
	if (_parameters.smoothing_ > 0.0)
		parent::_sumprodsolver.SetSmoothing(_parameters.smoothing_);
	else
		throw std::runtime_error("NesterovAcceleratedGradient::_InitSmoothing(): Error! Automatic smoothing selection is not implemented yet.");
};


template<class GM, class ACC>
template<class VISITOR>
InferenceTermination NesterovAcceleratedGradient<GM,ACC>::infer(VISITOR & vis)
{
	trws_base::VisitorWrapper<VISITOR,NesterovAcceleratedGradient<GM, ACC> > visitor(&vis,this);

	visitor.begin(parent::value(),parent::bound());

	if (parent::_sumprodsolver.GetSmoothing()<=0.0)
	{

	parent::_maxsumsolver.ForwardMove();
	parent::_maxsumsolver.EstimateIntegerLabelingAndBound();
	parent::_SelectOptimalBoundsAndLabeling();

	if (parent::_sumprodsolver.CheckDualityGap(parent::value(),parent::bound()))
	{
	#ifdef TRWS_DEBUG_OUTPUT
		parent::_fout << "NesterovAcceleratedGradient::_CheckStoppingCondition(): Precision attained! Problem solved!"<<std::endl;
	#endif

		 return NORMAL;
	}

	parent::_EstimateStartingSmoothing(visitor);

//	if (parent::_sumprodsolver.GetSmoothing()==std::numeric_limits<ValueType>::infinity())//DEBUG
//	{
//		hdf5::save(parent::graphicalModel(),"test-fail-nesterov.h5","gm");
//		return NORMAL;
//	}

	}else
	{
		parent::_sumprodsolver.SetSmoothing(_parameters.startSmoothingValue());
	}

	DDVectorType gradient(_currentDualVector.size()),
			lambda(_currentDualVector.size()),
			y(_currentDualVector),
			v(_currentDualVector);
	DDVectorType w(_currentDualVector.size());//temp variable
	ValueType   alpha,
	gamma= 1e6,//TODO: make it parameter _parameters.gamma0_,
	omega=_estimateOmega0();

	omega=omega/2.0;

	for (size_t i=0;i<_parameters.maxNumberOfIterations();++i)
	{
		parent::_fout <<"i="<<i<<std::endl;
		//gradient step with approximate linear search:
		ValueType doubledLipschitzConstant=2*getLipschitzConstant();//depends on a smoothing value
//===================== begin of internal loop ===========================================
		for (size_t j=0;j<_parameters.numberOfInternalIterations();++j)
		{
		ValueType mul=1.0;
		ValueType oldObjVal=_evaluateGradient(y,&gradient);
		parent::_fout <<"Dual smooth objective ="<<oldObjVal<<std::endl;
		ValueType norm2=std::inner_product(gradient.begin(),gradient.end(),gradient.begin(),(ValueType)0);//squared L2 norm
//		parent::_fout <<"squared gradient l-2 norm ="<<norm2<<std::endl;
		ValueType newObjVal;

		omega/=4.0;
		do
		{
			omega*=2.0;

			ACC::iop(-1.0,1.0,mul);
			std::transform(gradient.begin(),gradient.end(),lambda.begin(),std::bind1st(std::multiplies<ValueType>(),mul/omega));//!>lambda=y+gradient/omega; plus/minus depending on ACC
			std::transform(y.begin(),y.end(),lambda.begin(),lambda.begin(),std::plus<ValueType>());

			newObjVal=_evaluateSmoothObjective(lambda,((j+1)==_parameters.numberOfInternalIterations()));
		}
		while ( ACC::bop(newObjVal,(ValueType)(oldObjVal+mul*norm2/2.0/omega)) && (omega < doubledLipschitzConstant));//TODO: +/- and >/< depending on ACC

		if (omega >= doubledLipschitzConstant)
		{
#ifdef TRWS_DEBUG_OUTPUT
			parent::_fout << "Step size is smaller then the inverse Lipschitz constant. Passing to smoothing update." <<std::endl;
#endif
		}

		//if (!_parameters.plaingradient_)//TODO: make it parameter
		if(true)
		{
			//updating parameters
			alpha=(sqrt(gamma*gamma+4*omega*gamma)-gamma)/omega/2.0;
			gamma*=(1-alpha);
			//v+=(alpha/gamma)*gradient;
			trws_base::transform_inplace(gradient.begin(),gradient.end(),std::bind1st(std::multiplies<ValueType>(),mul*alpha/gamma));//!> plus/minus depending on ACC
			std::transform(v.begin(),v.end(),gradient.begin(),v.begin(),std::plus<ValueType>());

			//y=alpha*v+(1-alpha)*lambda;
			trws_base::transform_inplace(lambda.begin(),lambda.end(),std::bind1st(std::multiplies<ValueType>(),(1-alpha)));
			std::transform(v.begin(),v.end(),w.begin(),std::bind1st(std::multiplies<ValueType>(),alpha));
			std::transform(w.begin(),w.end(),lambda.begin(),y.begin(),std::plus<ValueType>());
		}else //plain gradient algorithm
		{
			std::copy(lambda.begin(),lambda.end(),y.begin());
		}
		}
//=================================== end of internal loop ===============================================
			parent::_maxsumsolver.ForwardMove();//initializes a move, makes a forward move and computes the dual bound, is used also in derivative computation in the next line
			parent::_maxsumsolver.EstimateIntegerLabelingAndBound();
#ifdef TRWS_DEBUG_OUTPUT
			parent::_fout << "_maxsumsolver.bound()=" <<parent::_maxsumsolver.bound()<<", _maxsumsolver.value()=" <<parent::_maxsumsolver.value() <<std::endl;
#endif

			ValueType  derivative=parent::_EstimateRhoDerivative();
#ifdef TRWS_DEBUG_OUTPUT
				parent::_fout << "derivative="<<derivative<<std::endl;
#endif

			InferenceTermination returncode;
			if ( parent::_CheckStoppingCondition(&returncode))
			{
				visitor.end(parent::value(), parent::bound());
				return NORMAL;
			}

			if( visitor(parent::value(),parent::bound()) != visitors::VisitorReturnFlag::ContinueInf ){
				break;
			}
			parent::_UpdateSmoothing(parent::_bestPrimalBound,parent::_maxsumsolver.bound(),parent::_sumprodsolver.bound(),derivative,i+1);


	}
	//update smoothing
	parent::_SelectOptimalBoundsAndLabeling();
	visitor.end(parent::value(), parent::bound());

	return NORMAL;
}


}//namespace opengm

#endif /* SMOOTH_NESTEROV_HXX_ */
