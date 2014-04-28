/*
 * smoothing_strategy.hxx
 *
 *  Created on: Jan 7, 2014
 *      Author: bsavchyn
 */

#ifndef SMOOTHING_STRATEGY_HXX_
#define SMOOTHING_STRATEGY_HXX_
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_reparametrization.hxx>

namespace opengm{
namespace trws_base{

template<class VALUETYPE>
struct SmoothingParameters
{
    typedef VALUETYPE ValueType;
    typedef enum {ADAPTIVE_DIMINISHING,WC_DIMINISHING,ADAPTIVE_PRECISIONORIENTED,WC_PRECISIONORIENTED} SmoothingStrategyType;

	static SmoothingStrategyType getSmoothingStrategyType(const std::string& name)
	{
		   if (name.compare("WC_DIMINISHING")==0) return WC_DIMINISHING;
		   else if (name.compare("ADAPTIVE_PRECISIONORIENTED")==0)  return ADAPTIVE_PRECISIONORIENTED;
		   else if (name.compare("WC_PRECISIONORIENTED")==0)  return WC_PRECISIONORIENTED;
		   else return ADAPTIVE_DIMINISHING;
	}

	static std::string getString(SmoothingStrategyType strategytype)
	{
		switch (strategytype)
		{
		case ADAPTIVE_DIMINISHING: return std::string("ADAPTIVE_DIMINISHING");
		case WC_DIMINISHING   : return std::string("WC_DIMINISHING");
		case ADAPTIVE_PRECISIONORIENTED   : return std::string("ADAPTIVE_PRECISIONORIENTED");
		case WC_PRECISIONORIENTED   : return std::string("WC_PRECISIONORIENTED");
		default: return std::string("UNKNOWN");
		}
	}

    SmoothingParameters(ValueType smoothingGapRatio=4,
    		           ValueType smoothingValue=0.0,
    		           ValueType smoothingDecayMultiplier=-1.0,
    		           ValueType precision=0,
    		           SmoothingStrategyType smoothingStrategy=ADAPTIVE_DIMINISHING):
    			smoothingGapRatio_(smoothingGapRatio),
    			smoothingValue_(smoothingValue),
    			smoothingDecayMultiplier_(smoothingDecayMultiplier),
    			precision_(precision),
    			smoothingStrategy_(smoothingStrategy){};

	ValueType smoothingGapRatio_;
	ValueType smoothingValue_;
	ValueType smoothingDecayMultiplier_;
	ValueType precision_;
	SmoothingStrategyType smoothingStrategy_;
	//bool worstCaseSmoothing_;
};





template<class VALUETYPE,class GM>
struct SmoothingBasedInference_Parameter_Base
{
	typedef VALUETYPE ValueType;
	typedef DecompositionStorage<GM> Storage;
	typedef SmoothingParameters<ValueType> SmoothingParametersType;
	typedef SumProdTRWS_Parameters<ValueType> SumProdSolverParametersType;
	typedef MaxSumTRWS_Parameters<ValueType> MaxSumSolverParametersType;
	typedef PrimalLPBound_Parameter<ValueType> PrimalLPEstimatorParametersType;

	SmoothingBasedInference_Parameter_Base(typename Storage::StructureType decompositionType,
									  bool lazyLPPrimalBoundComputation,
			 	 	 	 	 	 	  const SmoothingParametersType& smoothingParameters,
			 	 	 	 	 	      const SumProdSolverParametersType& sumProdSolverParameters,
			 	 	 	 	 	 	  const MaxSumSolverParametersType& maxSumSolverParameters,
									  const PrimalLPEstimatorParametersType& primalLPEstimatorParameters):
										  decompositionType_(decompositionType),
										  lazyLPPrimalBoundComputation_(lazyLPPrimalBoundComputation),
										  smoothingParameters_(smoothingParameters),
										  sumProdSolverParameters_(sumProdSolverParameters),
										  maxSumSolverParameters_(maxSumSolverParameters),
										  primalLPEstimatorParameters_(primalLPEstimatorParameters)
	{};
	virtual ~SmoothingBasedInference_Parameter_Base(){};

	typename Storage::StructureType decompositionType_;
	bool lazyLPPrimalBoundComputation_;
	const SmoothingParametersType& getSmoothingParameters()const{return smoothingParameters_;}
	const SumProdSolverParametersType& getSumProdSolverParameters()const{return sumProdSolverParameters_;}
	const MaxSumSolverParametersType& getMaxSumSolverParameters()const{return maxSumSolverParameters_;}
	const PrimalLPEstimatorParametersType& getPrimalLPEstimatorParameters()const{return primalLPEstimatorParameters_;}
protected:
	SmoothingParametersType smoothingParameters_;
	SumProdSolverParametersType sumProdSolverParameters_;
	MaxSumSolverParametersType maxSumSolverParameters_;
	PrimalLPEstimatorParametersType primalLPEstimatorParameters_;
};

template<class ValueType,class GM>
struct SmoothingBasedInference_Parameter : public trws_base::SmoothingBasedInference_Parameter_Base<ValueType,GM>

{
	typedef trws_base::DecompositionStorage<GM> Storage;
	typedef typename trws_base::SmoothingBasedInference_Parameter_Base<ValueType,GM> parent;
	typedef typename parent::SmoothingParametersType SmoothingParametersType;
	typedef typename parent::SumProdSolverParametersType SumProdSolverParametersType;
	typedef typename parent::MaxSumSolverParametersType MaxSumSolverParametersType;
	typedef typename parent::PrimalLPEstimatorParametersType PrimalLPEstimatorParametersType;
	typedef typename SmoothingParametersType::SmoothingStrategyType SmoothingStrategyType;

	SmoothingBasedInference_Parameter(size_t numOfExternalIterations=0,
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
			    SmoothingStrategyType smoothingStrategy=SmoothingParametersType::ADAPTIVE_DIMINISHING,
//			    bool worstCaseSmoothing=false,
			    bool fastComputations=true,
			    bool verbose=false
			    )
	 :parent(decompositionType,
			 lazyLPPrimalBoundComputation,
			 SmoothingParametersType(smoothingGapRatio,startSmoothingValue,smoothingDecayMultiplier,precision,smoothingStrategy),
			 //SumProdSolverParametersType(numOfInternalIterations,startSmoothingValue,precision,absolutePrecision,std::numeric_limits<ValueType>::epsilon(),fastComputations,canonicalNormalization),
			 SumProdSolverParametersType(numOfInternalIterations,startSmoothingValue,precision,absolutePrecision,0.0,fastComputations,canonicalNormalization),
			 MaxSumSolverParametersType(presolveMaxIterNumber,precision,absolutePrecision,presolveMinRelativeDualImprovement,fastComputations,canonicalNormalization),
			 PrimalLPEstimatorParametersType(primalBoundPrecision,maxPrimalBoundIterationNumber)
			 ),
	  numOfExternalIterations_(numOfExternalIterations),
	  verbose_(verbose)
	  {};

	  size_t numOfExternalIterations_;
	  bool verbose_;

	  /*
	   * Main algorithm parameters
	   */
	  size_t& maxNumberOfIterations(){return numOfExternalIterations_;}
	  const size_t& maxNumberOfIterations()const {return numOfExternalIterations_;}

	  size_t& numberOfInternalIterations(){return parent::sumProdSolverParameters_.maxNumberOfIterations_;}
	  const size_t& numberOfInternalIterations()const{return parent::sumProdSolverParameters_.maxNumberOfIterations_;}

//	  ValueType& precision(){return parent::sumProdSolverParameters_.precision_;}
	  const ValueType& precision()const{return parent::sumProdSolverParameters_.precision_;}
	  void setPrecision(ValueType precision){parent::maxSumSolverParameters_.precision_=parent::sumProdSolverParameters_.precision_=parent::smoothingParameters_.precision_=precision;}

	  bool&      isAbsolutePrecision(){return parent::sumProdSolverParameters_.absolutePrecision_;}
	  const bool&      isAbsolutePrecision()const {return parent::sumProdSolverParameters_.absolutePrecision_;}

	  ValueType& smoothingGapRatio(){return parent::smoothingParameters_.smoothingGapRatio_;}
	  const ValueType& smoothingGapRatio()const{return parent::smoothingParameters_.smoothingGapRatio_;}

	  bool& lazyLPPrimalBoundComputation(){return parent::lazyLPPrimalBoundComputation_;}
	  const bool& lazyLPPrimalBoundComputation()const{return parent::lazyLPPrimalBoundComputation_;}

//	  bool& lazyDerivativeComputation(){return lazyDerivativeComputation_;}
//	  const bool& lazyDerivativeComputation()const {return lazyDerivativeComputation_;}

      ValueType& smoothingDecayMultiplier(){return parent::smoothingParameters_.smoothingDecayMultiplier_;}
	  const ValueType& smoothingDecayMultiplier()const{return parent::smoothingParameters_.smoothingDecayMultiplier_;}

//	  bool& worstCaseSmoothing(){return parent::smoothingParameters_.worstCaseSmoothing_;}
//	  const bool& worstCaseSmoothing()const{return parent::smoothingParameters_.worstCaseSmoothing_;}
	  SmoothingStrategyType& smoothingStrategy(){return parent::smoothingParameters_.smoothingStrategy_;}
	  const SmoothingStrategyType& smoothingStrategy()const{return parent::smoothingParameters_.smoothingStrategy_;}

	  typename Storage::StructureType& decompositionType(){return parent::decompositionType_;}
	  const typename Storage::StructureType& decompositionType()const{return parent::decompositionType_;}

	  const ValueType& startSmoothingValue()const{return parent::sumProdSolverParameters_.smoothingValue_;}
	  void setStartSmoothingValue(ValueType val){ parent::smoothingParameters_.smoothingValue_=parent::sumProdSolverParameters_.smoothingValue_=val;}//TODO: duplicated!

	  const bool& fastComputations()const{return parent::sumProdSolverParameters_.fastComputations_;}
	  void setFastComputations(bool fast){parent::sumProdSolverParameters_.fastComputations_=parent::maxSumSolverParameters_.fastComputations_=fast;}

	  const bool& canonicalNormalization()const{return parent::sumProdSolverParameters_.canonicalNormalization_;}
	  void setCanonicalNormalization(bool canonical){parent::sumProdSolverParameters_.canonicalNormalization_=parent::maxSumSolverParameters_.canonicalNormalization_=canonical;}

	  /*
	   * Presolve parameters
	   */
	  size_t& maxNumberOfPresolveIterations(){return parent::maxSumSolverParameters_.maxNumberOfIterations_;}
	  const size_t& maxNumberOfPresolveIterations()const{return parent::maxSumSolverParameters_.maxNumberOfIterations_;}

	  ValueType& presolveMinRelativeDualImprovement() {return parent::maxSumSolverParameters_.minRelativeDualImprovement_;}
	  const ValueType& presolveMinRelativeDualImprovement()const {return parent::maxSumSolverParameters_.minRelativeDualImprovement_;}

	  /*
	   * Fractional primal bound estimator parameters
	   */
	  size_t& maxPrimalBoundIterationNumber(){return parent::primalLPEstimatorParameters_.maxIterationNumber_;}
	  const size_t& maxPrimalBoundIterationNumber()const{return parent::primalLPEstimatorParameters_.maxIterationNumber_;}

	  ValueType& primalBoundRelativePrecision(){return parent::primalLPEstimatorParameters_.relativePrecision_;}
	  const ValueType& primalBoundRelativePrecision()const{return parent::primalLPEstimatorParameters_.relativePrecision_;}

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
		  fout <<"smoothingDecayMultiplier=" << smoothingDecayMultiplier()<< std::endl;

		  fout << "smoothing strategy="<<SmoothingParametersType::getString(smoothingStrategy())<<std::endl;
		  fout << "decompositionType=" << Storage::getString(decompositionType()) << std::endl;

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



//====================================================================================
template<class GM, class ACC>
class SmoothingBasedInference;

template<class GM,class ACC>
class SmoothingStrategy
{
public:
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef SmoothingParameters<ValueType> Parameter;

	SmoothingStrategy(ValueType smoothingMultiplier, const Parameter& param=Parameter(), std::ostream& fout=std::cout):
		_fout(fout),
		_initializationStage(true),
		_smoothingMultiplier(smoothingMultiplier),
		_parameters(param),
		_oracleCallsCounter(0){};
	virtual ~SmoothingStrategy(){};
	virtual ValueType InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
			ValueType primalBound,
			ValueType dualBound){
		_initializationStage = false;
		_oracleCallsCounter=1;
		return 0;
	};
	virtual ValueType UpdateSmoothing(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			ValueType smoothingDerivative,
			size_t iterationCounter)=0;
	virtual bool SmoothingMustBeDecreased(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			size_t iterationCounter)=0;
	template<class DualDecompositionStorage >
	static ValueType ComputeSmoothingMultiplier(const DualDecompositionStorage& storage)
	{
		//compute max number of labels
		typename DualDecompositionStorage::LabelType numOfLabels=0;
		for (size_t i=0;i<storage.numberOfSharedVariables();++i)
			numOfLabels=std::max(numOfLabels,storage.numberOfLabels(i));

		//multiplier = \sum_{i=1}^n\log|\SX_i| <= \sum_{i=1}^n numOfLabels
		ValueType multiplier=0;
		ValueType logLabels=log((ValueType)numOfLabels);
		for (size_t i=0;i<storage.numberOfModels();++i)
			multiplier+=storage.size(i)*logLabels;

		return multiplier;
	}

	size_t getOracleCallsCounter()const{return _oracleCallsCounter;}
protected:
	ValueType getWorstCaseSmoothing(){return _parameters.precision_/2.0/_smoothingMultiplier;}
	std::ostream& _fout;
	bool _initializationStage;
	ValueType _smoothingMultiplier;
	Parameter _parameters;
	size_t _oracleCallsCounter;
};

template<class GM,class ACC>
class WorstCaseDiminishingSmoothing : public SmoothingStrategy<GM,ACC>
{
public:
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef SmoothingParameters<ValueType> Parameter;
	typedef SmoothingStrategy<GM,ACC> parent;

	WorstCaseDiminishingSmoothing(ValueType smoothingMultiplier,const Parameter& param=Parameter(), std::ostream& fout=std::cout):parent(smoothingMultiplier,param,fout){};

	ValueType InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
			ValueType primalBound,
			ValueType dualBound);

	ValueType UpdateSmoothing(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			ValueType smoothingDerivative,
			size_t iterationCounter);

	bool SmoothingMustBeDecreased(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			size_t iterationCounter)
	{
		return (UpdateSmoothing(smoothingValue,primalBound,dualBound,smoothDualBound,0,iterationCounter)<smoothingValue);
	}

};
template<class GM,class ACC>
typename WorstCaseDiminishingSmoothing<GM,ACC>::ValueType
WorstCaseDiminishingSmoothing<GM,ACC>::InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
		ValueType primalBound,
		ValueType dualBound)
{
			ValueType smoothing = fabs((primalBound-dualBound)/parent::_smoothingMultiplier/(2.0*parent::_parameters.smoothingGapRatio_-1));
		#ifdef TRWS_DEBUG_OUTPUT
			parent::_fout <<"_maxsumsolver.value()="<<primalBound<<", _maxsumsolver.bound()="<<dualBound<<std::endl;
			parent::_fout << "WorstCaseSmoothing="<<smoothing<<std::endl;
		#endif

			ValueType derivativeValue;
			ValueType smoothDualBound=smoothInference.UpdateSmoothDualEstimates(smoothing,&derivativeValue);
			smoothing = UpdateSmoothing(smoothing,primalBound,dualBound,smoothDualBound,derivativeValue,0);

			parent::_initializationStage = false;
			parent::_oracleCallsCounter=1;
			return smoothing;
};


template<class GM,class ACC>
typename WorstCaseDiminishingSmoothing<GM,ACC>::ValueType
WorstCaseDiminishingSmoothing<GM,ACC>::UpdateSmoothing(ValueType smoothingValue,
		ValueType primalBound,
		ValueType dualBound,
		ValueType smoothDualBound,
		ValueType smoothingDerivative,
		size_t iterationCounter)
		{
			return fabs((primalBound-smoothDualBound)/parent::_smoothingMultiplier/(2.0*parent::_parameters.smoothingGapRatio_));
		}

template<class GM,class ACC>
class AdaptiveDiminishingSmoothing : public SmoothingStrategy<GM,ACC>
{
public:
	typedef SmoothingStrategy<GM,ACC> parent;
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef typename parent::Parameter Parameter;
	AdaptiveDiminishingSmoothing(ValueType smoothingMultiplier,const Parameter& param=Parameter(), std::ostream& fout=std::cout):parent(smoothingMultiplier,param,fout){};
	~AdaptiveDiminishingSmoothing(){};
	ValueType InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
			ValueType primalBound,
			ValueType dualBound);
	ValueType UpdateSmoothing(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			ValueType smoothingDerivative,
			size_t iterationCounter);
	bool SmoothingMustBeDecreased(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			size_t iterationCounter);
protected:
	std::pair<ValueType,ValueType>	_getSmoothingInequality(ValueType primalBound,ValueType dualBound,ValueType smoothDualBound)const;
 };

template<class GM,class ACC>
typename AdaptiveDiminishingSmoothing<GM,ACC>::ValueType
AdaptiveDiminishingSmoothing<GM,ACC>::
InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
		ValueType primalBound,
		ValueType dualBound)
{
	ValueType smoothing = fabs((primalBound-dualBound)/parent::_smoothingMultiplier/(2.0*parent::_parameters.smoothingGapRatio_-1));
#ifdef TRWS_DEBUG_OUTPUT
	parent::_fout <<"_maxsumsolver.value()="<<primalBound<<", _maxsumsolver.bound()="<<dualBound<<std::endl;
	parent::_fout << "WorstCaseSmoothing="<<smoothing<<std::endl;
#endif

	ValueType derivativeValue;
	ValueType smoothDualBound=smoothInference.UpdateSmoothDualEstimates(smoothing,&derivativeValue);

//	visitor(primalBound,dualBound);
	parent::_oracleCallsCounter=0;
	do{
	++parent::_oracleCallsCounter;
	parent::_parameters.smoothingGapRatio_*=2;//!> increase the ratio to obtain fulfillment of a smoothing changing condition
	smoothing=UpdateSmoothing(smoothing,primalBound,dualBound,smoothDualBound,derivativeValue,0);
	parent::_parameters.smoothingGapRatio_/=2;//!> restoring the normal value before checking the condition
	smoothDualBound=smoothInference.UpdateSmoothDualEstimates(smoothing,&derivativeValue);
	}while (SmoothingMustBeDecreased(smoothing,primalBound,dualBound,smoothDualBound,0));
//	visitor(primalBound,dualBound);
	parent::_initializationStage = false;
	return smoothing;
};



template<class GM,class ACC>
bool AdaptiveDiminishingSmoothing<GM,ACC>::
SmoothingMustBeDecreased(ValueType smoothingValue,
		ValueType primalBound,
		ValueType dualBound,
		ValueType smoothDualBound,
		size_t iterationCounter)
{
	std::pair<ValueType,ValueType> lhsRhs=_getSmoothingInequality(primalBound,dualBound,smoothDualBound);
	if (parent::_parameters.smoothingDecayMultiplier_ <= 0.0 ||parent:: _initializationStage)
	{
		return ACC::ibop(lhsRhs.first,lhsRhs.second);
	}
	else
		return true;
}

template<class GM,class ACC>
typename AdaptiveDiminishingSmoothing<GM,ACC>::ValueType
AdaptiveDiminishingSmoothing<GM,ACC>::UpdateSmoothing(ValueType smoothingValue,
		ValueType primalBound,
		ValueType dualBound,
		ValueType smoothDualBound,
		ValueType smoothingDerivative,
		size_t iterationCounter){
#ifdef TRWS_DEBUG_OUTPUT
	parent::_fout << "dualBound="<<dualBound<<", smoothDualBound="<<smoothDualBound<<", derivativeValue="<<smoothingDerivative<<std::endl;
#endif

	if (SmoothingMustBeDecreased(smoothingValue,primalBound,dualBound,smoothDualBound,iterationCounter) || parent::_initializationStage)
	{
		ValueType newsmoothing;
		std::pair<ValueType,ValueType> lhsRhs = _getSmoothingInequality(primalBound,dualBound,smoothDualBound);
		newsmoothing=smoothingValue - (lhsRhs.second - lhsRhs.first)*(2.0*parent::_parameters.smoothingGapRatio_)/(2.0*parent::_parameters.smoothingGapRatio_-1.0)/smoothingDerivative;

		if ( (parent::_parameters.smoothingDecayMultiplier_ > 0.0) && (!parent::_initializationStage) )
		{
			ValueType newMulIt=parent::_parameters.smoothingDecayMultiplier_*(iterationCounter+1)+1;
			ValueType oldMulIt=parent::_parameters.smoothingDecayMultiplier_*iterationCounter+1;
			newsmoothing=std::min(newsmoothing,smoothingValue*oldMulIt/newMulIt);
		}

		//if ((newsmoothing > 0) && (newsmoothing!=std::numeric_limits<ValueType>::infinity()))//DEBUG
		if (newsmoothing > 0)
			if ((newsmoothing < smoothingValue) ||  parent::_initializationStage)
			{
#ifdef TRWS_DEBUG_OUTPUT
				parent::_fout << "smoothing changed to " <<  newsmoothing<<std::endl;
#endif
				return newsmoothing;
			}
	}
	return smoothingValue;
};

template<class GM,class ACC>
typename std::pair<typename AdaptiveDiminishingSmoothing<GM,ACC>::ValueType,typename AdaptiveDiminishingSmoothing<GM,ACC>::ValueType>
AdaptiveDiminishingSmoothing<GM,ACC>::
_getSmoothingInequality(ValueType primalBound,ValueType dualBound,ValueType smoothDualBound)const
{
	std::pair<ValueType,ValueType> lhsRhs;
	lhsRhs.first=dualBound-smoothDualBound;
	lhsRhs.second=(primalBound-smoothDualBound)/(2*parent::_parameters.smoothingGapRatio_);
	return lhsRhs;
}

template<class GM,class ACC>
class WorstCasePrecisionOrientedSmoothing : public SmoothingStrategy<GM,ACC>
{
public:
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef SmoothingParameters<ValueType> Parameter;
	typedef SmoothingStrategy<GM,ACC> parent;

	WorstCasePrecisionOrientedSmoothing(ValueType smoothingMultiplier,const Parameter& param=Parameter(), std::ostream& fout=std::cout):parent(smoothingMultiplier,param,fout){};

	ValueType InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
			ValueType primalBound,
			ValueType dualBound){
		ValueType smoothing=parent::getWorstCaseSmoothing();
		parent::_initializationStage= false;
		parent::_oracleCallsCounter=1;
#ifdef TRWS_DEBUG_OUTPUT
		parent::_fout << "smoothing = "<<smoothing<<std::endl;
#endif
		return smoothing;
	}

	ValueType UpdateSmoothing(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			ValueType smoothingDerivative,
			size_t iterationCounter){return parent::getWorstCaseSmoothing();};

	bool SmoothingMustBeDecreased(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			size_t iterationCounter)
	{return (parent::getWorstCaseSmoothing() < smoothingValue);}

};

template<class GM,class ACC>
class AdaptivePrecisionOrientedSmoothing : public SmoothingStrategy<GM,ACC>
{
public:
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	OPENGM_GM_TYPE_TYPEDEFS;
	typedef SmoothingParameters<ValueType> Parameter;
	typedef SmoothingStrategy<GM,ACC> parent;

	AdaptivePrecisionOrientedSmoothing(ValueType smoothingMultiplier,const Parameter& param=Parameter(), std::ostream& fout=std::cout):parent(smoothingMultiplier,param,fout){};

	ValueType InitSmoothing(SmoothingBasedInference<GM,ACC>& smoothInference,
			ValueType primalBound,
			ValueType dualBound){

		ValueType smoothing;
		if (parent::_parameters.smoothingValue_ > 0 ) smoothing=parent::_parameters.smoothingValue_; //!> starting smoothing provided
		else
		{
			ValueType derivativeValue,smoothDualBound;
			smoothing=parent::getWorstCaseSmoothing();
//			parent::_fout << "WorstCaseSmoothing = "<<smoothing<<", dualBound="<<dualBound<<std::endl;
			do{
				smoothing*=2;
//				parent::_fout << "test smoothing = "<<smoothing<<std::endl;
				smoothDualBound=smoothInference.UpdateSmoothDualEstimates(smoothing,&derivativeValue);
//				parent::_fout <<"smoothDualBound="<<smoothDualBound<<std::endl;
			}while (!SmoothingMustBeDecreased(smoothing,primalBound,dualBound,smoothDualBound,0));
			smoothing/=2.0;
		}

#ifdef TRWS_DEBUG_OUTPUT
		parent::_fout << "InitSmoothing = "<<smoothing<<std::endl;
#endif
		parent::_initializationStage= false;
		return smoothing;
	}

	ValueType UpdateSmoothing(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			ValueType smoothingDerivative,
			size_t iterationCounter){

		if (SmoothingMustBeDecreased(smoothingValue,primalBound,dualBound,smoothDualBound,iterationCounter))
		{
#ifdef TRWS_DEBUG_OUTPUT
		 parent::_fout << "Smoothing decreased to = "<<smoothingValue/2.0<<std::endl;
#endif
		 return smoothingValue/2.0;
		}else
		  return smoothingValue;
	};

	bool SmoothingMustBeDecreased(ValueType smoothingValue,
			ValueType primalBound,
			ValueType dualBound,
			ValueType smoothDualBound,
			size_t iterationCounter)
	{
		return fabs(dualBound-smoothDualBound) > parent::_parameters.precision_/2;
	}

};

//==============================================================
template<class GM, class ACC>
class SmoothingBasedInference : public Inference<GM, ACC>
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

	  //typedef typename MaxSumSolver::ReparametrizerType ReparametrizerType;
	  typedef TRWS_Reparametrizer<Storage,ACC> ReparametrizerType;
	  typedef SmoothingStrategy<GM,ACC> SmoothingStrategyType;

	  typedef SmoothingBasedInference_Parameter<ValueType,GM> Parameter;

	  SmoothingBasedInference(const GraphicalModelType& gm, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			  ,std::ostream& fout=std::cout
#endif
		):
	  _parameters(param),
	  _storage(gm,param.decompositionType_),
	  _sumprodsolver(_storage,param.getSumProdSolverParameters()
#ifdef TRWS_DEBUG_OUTPUT
			  ,fout
#endif
			  ),
	  _maxsumsolver(_storage,param.getMaxSumSolverParameters()
#ifdef TRWS_DEBUG_OUTPUT
	         ,fout//fout
#endif
			  ),
	  _estimator(gm,param.getPrimalLPEstimatorParameters()),
#ifdef TRWS_DEBUG_OUTPUT
	  _fout(fout),
#endif
	  _bestPrimalLPbound(ACC::template neutral<ValueType>()),
	  _bestPrimalBound(ACC::template neutral<ValueType>()),
	  _bestDualBound(ACC::template ineutral<ValueType>()),
	  _bestIntegerBound(ACC::template neutral<ValueType>()),
	  _bestIntegerLabeling(_storage.masterModel().numberOfVariables(),0.0)
	  {
		  ValueType smoothingMultiplier=SmoothingStrategyType::ComputeSmoothingMultiplier(_storage);

		  if ((param.smoothingStrategy()==Parameter::SmoothingParametersType::ADAPTIVE_PRECISIONORIENTED) ||
				  (param.smoothingStrategy()==Parameter::SmoothingParametersType::WC_PRECISIONORIENTED))
			  if (!param.isAbsolutePrecision())
				  throw std::runtime_error("SmoothingBasedInference: Error: relative precision can be used only with diminishing smoothing.");

		  switch (param.smoothingStrategy())
		  {
		  case Parameter::SmoothingParametersType::ADAPTIVE_DIMINISHING:
			  psmoothingStrategy=new typename trws_base::AdaptiveDiminishingSmoothing<GM,ACC>(smoothingMultiplier,param.getSmoothingParameters()
#ifdef TRWS_DEBUG_OUTPUT
			  				  ,_fout
#endif
			  				  );
			   break;
		  case Parameter::SmoothingParametersType::WC_DIMINISHING:
			  psmoothingStrategy=new typename trws_base::WorstCaseDiminishingSmoothing<GM,ACC>(smoothingMultiplier,param.getSmoothingParameters()
#ifdef TRWS_DEBUG_OUTPUT
			  					  ,_fout
#endif
			  					  );
			  break;
		  case Parameter::SmoothingParametersType::ADAPTIVE_PRECISIONORIENTED:
			  if (_parameters.isAbsolutePrecision())
			    psmoothingStrategy=new typename trws_base::AdaptivePrecisionOrientedSmoothing<GM,ACC>(smoothingMultiplier,param.getSmoothingParameters()
#ifdef TRWS_DEBUG_OUTPUT
			  				  ,_fout
#endif
			  				  );
			  else
				  std::runtime_error("SmoothingBasedInference::SmoothingBasedInference(): Error! Relative precision can not be used with precision oriented smoothing!");
			  break;
		  case Parameter::SmoothingParametersType::WC_PRECISIONORIENTED:
			  if (_parameters.isAbsolutePrecision())
			   psmoothingStrategy=new typename trws_base::WorstCasePrecisionOrientedSmoothing<GM,ACC>(smoothingMultiplier,param.getSmoothingParameters()
#ifdef TRWS_DEBUG_OUTPUT
			  				  ,_fout
#endif
			  				  );
			  else
				  std::runtime_error("SmoothingBasedInference::SmoothingBasedInference(): Error! Relative precision can not be used with precision oriented smoothing!");
			  break;
		  default: throw std::runtime_error("SmoothingBasedInference: Error: Unknown smoothing strategy type");
		  };
	  }

	  virtual ~SmoothingBasedInference(){delete psmoothingStrategy;  }

	  InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	  {out = _bestIntegerLabeling;
	   return opengm::NORMAL;}
	  const GraphicalModelType& graphicalModel() const { return _storage.masterModel(); }

	  ValueType bound() const{return _bestDualBound;}
	  ValueType value() const{return _bestIntegerBound;}

	  ValueType UpdateSmoothDualEstimates(ValueType smoothingValue,ValueType* pderivativeValue);

	  void getTreeAgreement(std::vector<bool>& out,std::vector<LabelType>* plabeling=0,std::vector<std::vector<LabelType> >* ptreeLabelings=0){_maxsumsolver.getTreeAgreement(out,plabeling,ptreeLabelings);}
	  Storage& getDecompositionStorage(){return _storage;}
	  const typename MaxSumSolver::FactorProperties& getFactorProperties()const {return _maxsumsolver.getFactorProperties();}
//	  ReparametrizerType* getReparametrizer(const typename ReparametrizerType::Parameter& params=typename ReparametrizerType::Parameter())const
//	  {return _maxsumsolver.getReparametrizer(params);}

	  ReparametrizerType * getReparametrizer(const typename ReparametrizerType::Parameter& params=typename ReparametrizerType::Parameter())//const //TODO: make it constant
	   {return new ReparametrizerType(_storage,_maxsumsolver.getFactorProperties(),params);}

protected:
	  template<class VISITOR>
	  InferenceTermination _Presolve(VISITOR& visitor,size_t* piterCounter=0);
	  template<class VISITOR>
	  size_t _EstimateStartingSmoothing(VISITOR& visitor);//!> returns number of oracle calls;
	  bool _UpdateSmoothing(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound, ValueType derivativeValue,size_t iterationCounterPlus1=0);
	  ValueType _EstimateRhoDerivative()const;
	  ValueType _FastEstimateRhoDerivative()const{return (_sumprodsolver.bound()-_maxsumsolver.bound())/_sumprodsolver.GetSmoothing();}
	  void _UpdatePrimalEstimator();
	  void _SelectOptimalBoundsAndLabeling();
	  bool _isLPBoundComputed()const;
	  bool _CheckStoppingCondition(InferenceTermination*);

	  Parameter 			_parameters;
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

	SmoothingStrategyType* psmoothingStrategy;
	  /*
	   * optimization of computations
	   */
    typename SumProdSolver::OutputContainerType _marginalsTemp;
};

template<class GM,class ACC>
typename SmoothingBasedInference<GM,ACC>::ValueType
SmoothingBasedInference<GM,ACC>::UpdateSmoothDualEstimates(ValueType smoothingValue,ValueType* pderivativeValue)
{
	  _sumprodsolver.SetSmoothing(smoothingValue);
	  _sumprodsolver.ForwardMove();
	  _sumprodsolver.GetMarginalsAndDerivativeMove();
	  *pderivativeValue=_EstimateRhoDerivative();
	  return _sumprodsolver.bound();
};

template<class GM,class ACC>
template<class VISITOR>
size_t SmoothingBasedInference<GM,ACC>::_EstimateStartingSmoothing(VISITOR& visitor)
{
	ValueType smoothingValue= psmoothingStrategy->InitSmoothing(*this,_maxsumsolver.value(),_maxsumsolver.bound());
	_sumprodsolver.SetSmoothing(smoothingValue);
	return psmoothingStrategy->getOracleCallsCounter();
};

template<class GM,class ACC>
bool SmoothingBasedInference<GM,ACC>::_UpdateSmoothing(ValueType primalBound,ValueType dualBound, ValueType smoothDualBound, ValueType derivativeValue,size_t iterationCounterPlus1)
{
	   ValueType newSmoothing=psmoothingStrategy->UpdateSmoothing(_sumprodsolver.GetSmoothing(),primalBound,dualBound,smoothDualBound,derivativeValue,iterationCounterPlus1-1);
	   if (newSmoothing != _sumprodsolver.GetSmoothing())
	   	   {
	   		   _sumprodsolver.SetSmoothing(newSmoothing);
	   		   return true;
	   	   }
	   else
		   return false;
}

template<class GM,class ACC>
typename SmoothingBasedInference<GM,ACC>::ValueType
SmoothingBasedInference<GM,ACC>::_EstimateRhoDerivative()const
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

template<class GM,class ACC>
void SmoothingBasedInference<GM,ACC>::_UpdatePrimalEstimator()
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
};

template<class GM,class ACC>
void SmoothingBasedInference<GM,ACC>::_SelectOptimalBoundsAndLabeling()
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
opengm::InferenceTermination SmoothingBasedInference<GM,ACC>::_Presolve(VISITOR& visitor, size_t* piterCounter)
{
#ifdef TRWS_DEBUG_OUTPUT
	 _fout << "Running TRWS presolve..."<<std::endl;
#endif
	 return _maxsumsolver.infer_visitor_updates(visitor,piterCounter);
}

template<class GM,class ACC>
bool SmoothingBasedInference<GM,ACC>::_isLPBoundComputed()const
{
	return (!_parameters.lazyLPPrimalBoundComputation_ || !ACC::bop(_sumprodsolver.value(),_bestPrimalBound) );
}

template<class GM,class ACC>
bool SmoothingBasedInference<GM,ACC>::_CheckStoppingCondition(InferenceTermination* preturncode)
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
	  _fout << "SmoothingBasedInference::_CheckStoppingCondition(): Precision attained! Problem solved!"<<std::endl;
#endif
	 *preturncode=CONVERGENCE;
	 return true;
	}

	return false;
}
//====================================================



}//trws_base
}//opengm

#endif /* SMOOTHING_STRATEGY_HXX_ */
