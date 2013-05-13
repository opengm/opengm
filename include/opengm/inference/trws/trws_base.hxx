#ifndef TRWS_BASE_HXX_
#define TRWS_BASE_HXX_
#include <iostream>
#include <time.h>
#include <opengm/inference/trws/trws_decomposition.hxx>
#include <opengm/inference/trws/trws_subproblemsolver.hxx>
#include <functional>
#include <opengm/functions/view_fix_variables_function.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/visitors/visitor.hxx>

namespace trws_base{

template<class GM>
class DecompositionStorage
{
public:
	typedef SequenceStorage<GM> SubModel;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename MonotoneChainsDecomposition<GM>::SubVariable SubVariable;
	typedef typename MonotoneChainsDecomposition<GM>::SubVariableListType SubVariableListType;
	typedef typename SubModel::UnaryFactor UnaryFactor;
	typedef enum {GRIDSTRUCTURE, GENERALSTRUCTURE} StructureType;
	typedef VariableToFactorMapping<GM> VariableToFactorMap;

	DecompositionStorage(const GM& gm,StructureType structureType=GENERALSTRUCTURE);
	~DecompositionStorage();

	const GM& masterModel()const{return _gm;}
	LabelType numberOfLabels(IndexType varId)const{return _gm.numberOfLabels(varId);}
	IndexType numberOfModels()const{return (IndexType)_subModels.size();}
	IndexType numberOfSharedVariables()const{return (IndexType)_variableDecomposition.size();}
	SubModel& subModel(IndexType modelId){return *_subModels[modelId];}
	const SubModel& subModel(IndexType modelId)const{return *_subModels[modelId];}
	IndexType size(IndexType subModelId)const{return (IndexType)_subModels[subModelId]->size();}

	const SubVariableListType& getSubVariableList(IndexType varId)const{return _variableDecomposition[varId];}
	StructureType getStructureType()const{return _structureType;}
#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
	void PrintVariableDecompositionConsistency(std::ostream& fout)const;
#endif

private:
	void _InitSubModels();
	const GM& _gm;
	StructureType  _structureType;
	std::vector<SubModel*> _subModels;
	std::vector<SubVariableListType> _variableDecomposition;
	VariableToFactorMap _var2FactorMap;
};

template<class VISITOR, class INFERENCE_TYPE>
class VisitorWrapper
{
public:
	typedef VISITOR VisitorType;
	typedef INFERENCE_TYPE InferenceType;
	typedef typename InferenceType::ValueType ValueType;

	VisitorWrapper(VISITOR* pvisitor,INFERENCE_TYPE* pinference)
	:_pvisitor(pvisitor),
	 _pinference(pinference){};
	void begin(ValueType value,ValueType bound){_pvisitor->begin(*_pinference,value,bound);}
	void end(ValueType value,ValueType bound){_pvisitor->end(*_pinference,value,bound);}
	void operator() (ValueType value,ValueType bound){(*_pvisitor)(*_pinference,value,bound);}
private:
	VISITOR* _pvisitor;
	INFERENCE_TYPE* _pinference;
};

template<class ValueType>
struct TRWSPrototype_Parameters
{
	size_t maxNumberOfIterations_;
	ValueType precision_;
	bool absolutePrecision_;//true for absolute precision, false for relative w.r.t. dual value
	ValueType minRelativeDualImprovement_;
	bool fastComputations_;

	TRWSPrototype_Parameters(size_t maxIternum,
			                 ValueType precision=1.0,
			                 bool absolutePrecision=true,
			                 ValueType minRelativeDualImprovement=-1.0,
			                 bool fastComputations=true):
		maxNumberOfIterations_(maxIternum),
		precision_(precision),
		absolutePrecision_(absolutePrecision),
		minRelativeDualImprovement_(minRelativeDualImprovement),
		fastComputations_(fastComputations)
		{};
};

template<class GM>
class PreviousFactorTable
{
public:
	typedef typename GM::IndexType IndexType;
	typedef SequenceStorage<GM> Storage;
	typedef typename Storage::MoveDirection MoveDirection;
	struct FactorVarID
	{
		FactorVarID(){};
		FactorVarID(IndexType fID,IndexType vID,IndexType lID):
			factorId(fID),varId(vID),localId(lID){};

#ifdef TRWS_DEBUG_OUTPUT
		void print(std::ostream& out)const{out <<"("<<factorId<<","<<varId<<","<<localId<<"),";}
#endif

		IndexType factorId;
		IndexType varId;
		IndexType localId;//local index of varId
	};
	typedef std::vector<FactorVarID> FactorList;
	typedef typename FactorList::const_iterator const_iterator;

	PreviousFactorTable(const GM& gm);
	const_iterator begin(IndexType varId,MoveDirection md)const{return (md==Storage::Direct ? _forwardFactors[varId].begin() : _backwardFactors[varId].begin());}
	const_iterator end(IndexType varId,MoveDirection md)const{return (md==Storage::Direct ? _forwardFactors[varId].end() : _backwardFactors[varId].end());}
#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout);
#endif
private:
	std::vector<FactorList> _forwardFactors;
	std::vector<FactorList> _backwardFactors;
};

template<class GM>
PreviousFactorTable<GM>::PreviousFactorTable(const GM& gm):
_forwardFactors(gm.numberOfVariables()),
_backwardFactors(gm.numberOfVariables())
{
 std::vector<IndexType> varIDs(2);
 for (IndexType factorId=0;factorId<gm.numberOfFactors();++factorId)
 {
	switch (gm[factorId].numberOfVariables())
	{
	 case 1: break;
	 case 2:
		 gm[factorId].variableIndices(varIDs.begin());
		 if (varIDs[0] < varIDs[1])
		 {
			 _forwardFactors[varIDs[1]].push_back(FactorVarID(factorId,varIDs[0],0));
			 _backwardFactors[varIDs[0]].push_back(FactorVarID(factorId,varIDs[1],1));
		 }
		 else
		 {
			 _forwardFactors[varIDs[0]].push_back(FactorVarID(factorId,varIDs[1],1));
			 _backwardFactors[varIDs[1]].push_back(FactorVarID(factorId,varIDs[0],0));
		 }
		 break;
	 default: throw std::runtime_error("PreviousFactor::PreviousFactor: only the factors of order <=2 are supported!");
	}
 }
}

#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void PreviousFactorTable<GM>::PrintTestData(std::ostream& fout)
{
   fout << "Forward factors:"<<std::endl;
   for (size_t varId=0;varId<_forwardFactors.size();++varId)
   {
	  fout << "varId="<<varId<<", ";
	  for (size_t i=0;i<_forwardFactors[varId].size();++i)
	    _forwardFactors[varId][i].print(fout);
	  fout <<std::endl;
   }

   fout << "Backward factors:"<<std::endl;
   for (size_t varId=0;varId<_backwardFactors.size();++varId)
   {
	  fout << "varId="<<varId<<", ";
	  for (size_t i=0;i<_backwardFactors[varId].size();++i)
	    _backwardFactors[varId][i].print(fout);
	  fout <<std::endl;
   }
}
#endif

template <class SubSolver>
class TRWSPrototype
{
public:
	typedef typename SubSolver::GMType GM;//TODO: remove me
	typedef GM GraphicalModelType;
	typedef typename SubSolver::ACCType ACC;//TODO: remove me
	typedef ACC AccumulationType;
	typedef SubSolver SubSolverType;
	typedef FunctionParameters<GM> FactorProperties;
	typedef opengm::EmptyVisitor< TRWSPrototype<SubSolverType> >  EmptyVisitorParent;
	typedef VisitorWrapper<EmptyVisitorParent,TRWSPrototype<SubSolver>  > EmptyVisitorType;

	typedef typename SubSolver::const_iterators_pair const_marginals_iterators_pair;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef opengm::InferenceTermination InferenceTermination;
	typedef typename std::vector<ValueType> OutputContainerType;
	typedef typename OutputContainerType::iterator OutputIteratorType;//TODO: make a template parameter

	typedef TRWSPrototype_Parameters<ValueType> Parameters;

	typedef SequenceStorage<GM> SubModel;
	typedef DecompositionStorage<GM> Storage;
	typedef typename Storage::UnaryFactor UnaryFactor;

	TRWSPrototype(Storage& storage,const Parameters& params
#ifdef TRWS_DEBUG_OUTPUT
			,std::ostream& fout=std::cout
#endif
			);
	virtual ~TRWSPrototype();

	virtual ValueType GetBestIntegerBound()const{return _bestIntegerBound;};
	virtual ValueType value()const{return _bestIntegerBound;}
	virtual ValueType bound()const{return _dualBound;}
	virtual const std::vector<LabelType>& arg()const{return _bestIntegerLabeling;}

#ifdef TRWS_DEBUG_OUTPUT
	virtual void PrintTestData(std::ostream& fout)const;
#endif

	bool CheckDualityGap(ValueType primalBound,ValueType dualBound);
	virtual std::pair<ValueType,ValueType> GetMarginals(IndexType variable, OutputIteratorType begin){return std::make_pair((ValueType)0,(ValueType)0);};//!>returns "averaged" over subsolvers marginals
	void GetMarginalsMove();
	void BackwardMove();//optimization move, also estimates a primal bound

	ValueType getBound(size_t i)const{return _subSolvers[i]->GetObjectiveValue();}
	virtual InferenceTermination infer(){EmptyVisitorParent vis; EmptyVisitorType visitor(&vis,this);  return infer(visitor);};
	template<class VISITOR> InferenceTermination infer(VISITOR&);
	void ForwardMove();
	ValueType lastDualUpdate()const{return _lastDualUpdate;}

	InferenceTermination core_infer(){EmptyVisitorParent vis; EmptyVisitorType visitor(&vis,this);  return _core_infer(visitor);};
protected:
	void _EstimateIntegerLabeling();
	template <class VISITOR> InferenceTermination _core_infer(VISITOR&);
	virtual ValueType _GetPrimalBound(){_EvaluateIntegerBounds(); return GetBestIntegerBound();}
	virtual void _postprocessMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end)=0;
	virtual void _normalizeMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end,SubSolver* subSolver)=0;
	void _EvaluateIntegerBounds();

	/*
	 * Integer labeling computation functions
	 */
	virtual void _SumUpForwardMarginals(std::vector<ValueType>* pout,const_marginals_iterators_pair itpair)=0;
	void _EstimateIntegerLabel(IndexType varId,const std::vector<ValueType>& sumMarginal)
	{_integerLabeling[varId]=std::max_element(sumMarginal.begin(),sumMarginal.end(),ACC::template ibop<ValueType>)-sumMarginal.begin();}//!>best label index

	void _InitSubSolvers();
	void _ForwardMove();
	void _FinalizeMove();
	ValueType _GetObjectiveValue();
	IndexType _order(IndexType i);
	IndexType _core_order(IndexType i,IndexType totalSize);
	bool _CheckConvergence(ValueType relativeThreshold);
	virtual bool _CheckStoppingCondition(InferenceTermination* pterminationCode);
	virtual void _EstimateTRWSBound(){};

	virtual void _InitMove()=0;

	Storage&    _storage;
	FactorProperties _factorProperties;
	PreviousFactorTable<GM> _ftable;
	Parameters _parameters;

#ifdef TRWS_DEBUG_OUTPUT
	std::ostream& _fout;
#endif

	ValueType _dualBound;//!>current dual bound (it is improved monotonically)
	ValueType _oldDualBound;//!> previous dual bound (it is improved monotonically)
	ValueType _lastDualUpdate;

	typename SubModel::MoveDirection _moveDirection;
	std::vector<SubSolver*> _subSolvers;

	std::vector<std::vector<ValueType> > _marginals;//!<computation optimization

	ValueType _integerBound;
	ValueType _bestIntegerBound;

	std::vector<LabelType> _integerLabeling;
	std::vector<LabelType> _bestIntegerLabeling;

	/* Computation optimization */
	std::vector<ValueType> _sumMarginal;
	mutable typename FactorProperties::ParameterStorageType _factorParameters;

private:
	TRWSPrototype(TRWSPrototype&);
	TRWSPrototype& operator =(TRWSPrototype&);
};

template<class ValueType>
struct SumProdTRWS_Parameters : public TRWSPrototype_Parameters<ValueType>
{
	typedef TRWSPrototype_Parameters<ValueType> parent;
	ValueType smoothingValue_;
	SumProdTRWS_Parameters(size_t maxIternum,
			   ValueType smValue,
			   ValueType precision=1.0,
			   bool absolutePrecision=true,
			   ValueType minRelativeDualImprovement=2*std::numeric_limits<ValueType>::epsilon(),
			   bool fastComputations=true)
	:parent(maxIternum,precision,absolutePrecision,minRelativeDualImprovement,fastComputations),
	 smoothingValue_(smValue){};
};

template<class GM,class ACC>
class SumProdTRWS : public TRWSPrototype<SumProdSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> >
{
public:
	typedef TRWSPrototype<SumProdSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> > parent;
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	typedef typename parent::SubSolverType SubSolver;
	typedef typename parent::const_marginals_iterators_pair const_marginals_iterators_pair;
	typedef typename parent::ValueType ValueType;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;
	typedef typename parent::InferenceTermination InferenceTermination;
	typedef SequenceStorage<GM> SubModel;
	typedef DecompositionStorage<GM> Storage;
	typedef typename parent::OutputContainerType OutputContainerType;
	typedef typename OutputContainerType::iterator OutputIteratorType;

	typedef SumProdTRWS_Parameters<ValueType> Parameters;

	SumProdTRWS(Storage& storage,const Parameters& params
#ifdef TRWS_DEBUG_OUTPUT
			,std::ostream& fout=std::cout
#endif
			):
		parent(storage,params
#ifdef TRWS_DEBUG_OUTPUT
				,fout
#endif
		),
		_smoothingValue(params.smoothingValue_)
		{};
	~SumProdTRWS(){};

#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
#endif

	void SetSmoothing(ValueType smoothingValue){_smoothingValue=smoothingValue;_InitMove();}
	ValueType GetSmoothing()const{return _smoothingValue;}
	/*
	 * returns "averaged" over subsolvers marginals
	 * and pair of (ell_2 norm,ell_infty norm)
	 */
	std::pair<ValueType,ValueType> GetMarginals(IndexType variable, OutputIteratorType begin);
	ValueType GetMarginalsAndDerivativeMove();//!> besides computation of marginals returns derivative w.r.t. _smoothingValue
	ValueType getDerivative(size_t i)const{return parent::_subSolvers[i]->getDerivative();}

protected:
	void _SumUpForwardMarginals(std::vector<ValueType>* pout,const_marginals_iterators_pair itpair);
	void _postprocessMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end);
	void _normalizeMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end,SubSolver* subSolver);
	void _InitMove();
	//bool _CheckConvergence();
	//bool _CheckStoppingCondition(InferenceTermination* pterminationCode);
	ValueType _smoothingValue;
};

template<class ValueType>
struct MaxSumTRWS_Parameters : public TRWSPrototype_Parameters<ValueType>
{
	typedef TRWSPrototype_Parameters<ValueType> parent;
	MaxSumTRWS_Parameters(size_t maxIternum,
			   ValueType precision=1.0,
			   bool absolutePrecision=true,
			   ValueType minRelativeDualImprovement=-1.0,
			   bool fastComputations=true,
			   bool canonicalNormalization=false):
		parent(maxIternum,precision,absolutePrecision,minRelativeDualImprovement,fastComputations),
		canonicalNormalization_(canonicalNormalization){};

	bool canonicalNormalization_;
};

template<class GM,class ACC>
class MaxSumTRWS : public TRWSPrototype<MaxSumSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> >
{
public:
	typedef TRWSPrototype<MaxSumSolver<GM,ACC,typename std::vector<typename GM::ValueType>::const_iterator> > parent;
	//typedef typename parent::Parameters Parameters;
	typedef typename parent::SubSolverType SubSolver;
	typedef typename parent::const_marginals_iterators_pair const_marginals_iterators_pair;
	typedef typename parent::ValueType ValueType;
	typedef typename parent::InferenceTermination InferenceTermination;
	typedef typename parent::EmptyVisitorType EmptyVisitorType;
	typedef typename parent::UnaryFactor UnaryFactor;
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;
	typedef typename parent::OutputContainerType OutputContainerType;

	typedef SequenceStorage<GM> SubModel;
	typedef DecompositionStorage<GM> Storage;

	typedef MaxSumTRWS_Parameters<ValueType> Parameters;

	MaxSumTRWS(Storage& storage,const Parameters& params
#ifdef TRWS_DEBUG_OUTPUT
			,std::ostream& fout=std::cout
#endif
	):
		parent(storage,params
#ifdef TRWS_DEBUG_OUTPUT
				,fout
#endif
				),
		_canonicalNormalization(params.canonicalNormalization_),
		_pseudoBoundValue(0.0),
		_localConsistencyCounter(0)
	{}
	~MaxSumTRWS(){};

	//InferenceTermination infer();
	void getTreeAgreement(std::vector<bool>& out);
	bool CheckTreeAgreement(InferenceTermination* pterminationCode);
protected:
	void _SumUpForwardMarginals(std::vector<ValueType>* pout,const_marginals_iterators_pair itpair);
	void _postprocessMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end);
	void _normalizeMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end,SubSolver* subSolver);
	void _InitMove();
	void _EstimateTRWSBound();
	bool _CheckStoppingCondition(InferenceTermination* pterminationCode);

	bool _canonicalNormalization;
	ValueType _pseudoBoundValue;
	size_t _localConsistencyCounter;
	/*
	 * computaton optimization
	 */
	std::vector<bool> _treeAgree;
	std::vector<bool> _mask;
	std::vector<bool> _nodeMask;
};
//============ TRWSPrototype IMPLEMENTATION ======================================

template <class SubSolver>
TRWSPrototype<SubSolver>::TRWSPrototype(Storage& storage,const Parameters& params
#ifdef TRWS_DEBUG_OUTPUT
		,std::ostream& fout
#endif
):
_storage(storage),
_factorProperties(storage.masterModel()),
_ftable(storage.masterModel()),
_parameters(params),
#ifdef TRWS_DEBUG_OUTPUT
_fout(fout),
#endif
_dualBound(ACC::template ineutral<ValueType>()),
_oldDualBound(ACC::template ineutral<ValueType>()),
_lastDualUpdate(0),
_moveDirection(SubModel::Direct),
_subSolvers(),
_marginals(),
_integerBound(ACC::template neutral<ValueType>()),
_bestIntegerBound(ACC::template neutral<ValueType>()),
_integerLabeling(storage.masterModel().numberOfVariables(),0),
_bestIntegerLabeling(storage.masterModel().numberOfVariables(),0),
_sumMarginal()
{
#ifdef TRWS_DEBUG_OUTPUT
	_fout.precision(16);
#endif
	_InitSubSolvers();
	_marginals.resize(_storage.numberOfModels());
#ifdef TRWS_DEBUG_OUTPUT
	_factorProperties.PrintStatusData(fout);
#endif
}

template <class SubSolver>
TRWSPrototype<SubSolver>::~TRWSPrototype()
{
	for_each(_subSolvers.begin(),_subSolvers.end(),DeallocatePointer<SubSolver>);
};

template <class SubSolver>
void TRWSPrototype<SubSolver>::_InitSubSolvers()
{
	_subSolvers.resize(_storage.numberOfModels());
	for (size_t modelId=0;modelId<_subSolvers.size();++modelId)
		_subSolvers[modelId]= new SubSolver(_storage.subModel(modelId),_factorProperties,_parameters.fastComputations_);
}

template <class SubSolver>
bool TRWSPrototype<SubSolver>::CheckDualityGap(ValueType primalBound,ValueType dualBound)
{
	//TODO: check that primal bound > dualBound if (bop(primalBound,dualBound)

	if (_parameters.absolutePrecision_)
	{
		if (fabs(primalBound-dualBound) <= _parameters.precision_)
		{
			return true;
		}
	}
	else
	{
		if (fabs((primalBound-dualBound)/dualBound)<= _parameters.precision_ )
			return true;
	}
	return false;
}

template <class SubSolver>
bool TRWSPrototype<SubSolver>::_CheckConvergence(ValueType relativeThreshold)
{
	if (relativeThreshold >=0.0)
	{
	ValueType mul; ACC::iop(-1.0,1.0,mul);
	if (ACC::bop(_dualBound, (_oldDualBound + _dualBound*mul*relativeThreshold)))
		return true;
	}
	return false;
}

template <class SubSolver>
bool TRWSPrototype<SubSolver>::_CheckStoppingCondition(InferenceTermination* pterminationCode)
{
	_lastDualUpdate=fabs(_dualBound-_oldDualBound);

	if (CheckDualityGap(_bestIntegerBound,_dualBound))
	{
#ifdef TRWS_DEBUG_OUTPUT
		_fout << "TRWSPrototype::_CheckStoppingCondition(): duality gap <= specified precision!" <<std::endl;
#endif
		*pterminationCode=opengm::CONVERGENCE;
		return true;
	}

	if (_CheckConvergence(_parameters.minRelativeDualImprovement_))
	{
#ifdef TRWS_DEBUG_OUTPUT
		_fout << "TRWSPrototype::_CheckStoppingCondition(): Dual update is smaller than the specified threshold. Stopping"<<std::endl;
#endif
		*pterminationCode=opengm::NORMAL;
		return true;
	}

	_oldDualBound=_dualBound;

	return false;
}

template <class SubSolver>
template <class VISITOR>
typename TRWSPrototype<SubSolver>::InferenceTermination TRWSPrototype<SubSolver>::_core_infer(VISITOR& visitor)
{
	for (size_t iterationCounter=0;iterationCounter<_parameters.maxNumberOfIterations_;++iterationCounter)
	{
#ifdef TRWS_DEBUG_OUTPUT
		_fout <<"Iteration Nr."<<iterationCounter<<"-------------------------------------"<<std::endl;
#endif

		BackwardMove();

#ifdef TRWS_DEBUG_OUTPUT
		_fout << "dualBound=" << _dualBound <<", primalBound="<<_GetPrimalBound() <<std::endl;
#endif
		_EstimateTRWSBound();
		visitor(value(),bound());
		InferenceTermination returncode;
		if (_CheckStoppingCondition(&returncode))
			 return returncode;
	}
	return opengm::TIMEOUT;
}

template <class SubSolver>
typename TRWSPrototype<SubSolver>::ValueType TRWSPrototype<SubSolver>::_GetObjectiveValue()
{
	ValueType 	dualBound=0;
	for (size_t i=0;i<_subSolvers.size();++i)
		dualBound+=_subSolvers[i]->GetObjectiveValue();

	return dualBound;
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::_ForwardMove()
{
	std::for_each(_subSolvers.begin(), _subSolvers.end(), std::mem_fun(&SubSolver::Move));
	_moveDirection=SubModel::ReverseDirection(_moveDirection);
	_dualBound=_GetObjectiveValue();
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::GetMarginalsMove()
{
	std::for_each(_subSolvers.begin(), _subSolvers.end(), std::mem_fun(&SubSolver::MoveBack));
	_moveDirection=SubModel::ReverseDirection(_moveDirection);
}

template <class SubSolver>
typename TRWSPrototype<SubSolver>::IndexType TRWSPrototype<SubSolver>::_core_order(IndexType i,IndexType totalSize)
{
	return (_moveDirection==SubModel::Direct ? i : totalSize-i-1);
}

template <class SubSolver>
typename TRWSPrototype<SubSolver>::IndexType TRWSPrototype<SubSolver>::_order(IndexType i)
{
	return _core_order(i,_storage.numberOfSharedVariables());
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::_FinalizeMove()
{
	std::for_each(_subSolvers.begin(), _subSolvers.end(), std::mem_fun(&SubSolver::FinalizeMove));
	_moveDirection=SubModel::ReverseDirection(_moveDirection);
	_EstimateIntegerLabeling();
}

#ifdef TRWS_DEBUG_OUTPUT
template <class SubSolver>
void TRWSPrototype<SubSolver>::PrintTestData(std::ostream& fout)const
{
	fout << "_dualBound:" << _dualBound <<std::endl;
	fout << "_oldDualBound:" << _oldDualBound <<std::endl;
	fout << "_lastDualUpdate=" << _lastDualUpdate << std::endl;
	fout << "_moveDirection:" << _moveDirection <<std::endl;
	fout << "_integerBound=" << _integerBound << std::endl;
	fout << "_bestIntegerBound=" << _bestIntegerBound << std::endl;
	fout << "_integerLabeling=" << _integerLabeling;
	fout << "_bestIntegerLabeling=" << _bestIntegerLabeling;
}
#endif

template <class SubSolver>
template <class VISITOR>
typename TRWSPrototype<SubSolver>::InferenceTermination TRWSPrototype<SubSolver>::infer(VISITOR& visitor)
{
	_InitMove();
	_ForwardMove();
	_oldDualBound=_dualBound;
	visitor.begin(value(),bound());
#ifdef TRWS_DEBUG_OUTPUT
	_fout << "ForwardMove: dualBound=" << _dualBound <<std::endl;
#endif
	InferenceTermination returncode;
	returncode=_core_infer(visitor);
	visitor.end(value(), bound());
	return returncode;
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::ForwardMove()
{
	_InitMove();
	_ForwardMove();
	_dualBound=_GetObjectiveValue();
}


template <class SubSolver>
void TRWSPrototype<SubSolver>::BackwardMove()
{
	std::vector<ValueType> averageMarginal;

	for (IndexType i=0;i<_storage.numberOfSharedVariables();++i)
	{
		IndexType varId=_order(i);
		const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);
		averageMarginal.assign(_storage.numberOfLabels(varId),0.0);

		//<!computing average marginals
		for(typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();modelIt!=varList.end();++modelIt)
		{
			SubSolver& subSolver=*_subSolvers[modelIt->subModelId_];
			std::vector<ValueType>& marginals=_marginals[modelIt->subModelId_];
			marginals.resize(_storage.numberOfLabels(varId));

			IndexType startNodeIndex=_core_order(0,_storage.size(modelIt->subModelId_));

			if (modelIt->subVariableId_!=startNodeIndex)
				subSolver.PushBack();

			typename SubSolver::const_iterators_pair marginalsit=subSolver.GetMarginals();

			std::copy(marginalsit.first,marginalsit.second,marginals.begin());
			_normalizeMarginals(marginals.begin(),marginals.end(),&subSolver);
			std::transform(marginals.begin(),marginals.end(),averageMarginal.begin(),averageMarginal.begin(),std::plus<ValueType>());
		}
		transform_inplace(averageMarginal.begin(),averageMarginal.end(),std::bind1st(std::multiplies<ValueType>(),-1.0/varList.size()));


		//<!reweighting submodels

		for(typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();modelIt!=varList.end();++modelIt)
		{
			SubSolver& subSolver=*_subSolvers[modelIt->subModelId_];
			std::vector<ValueType>& marginals=_marginals[modelIt->subModelId_];

			std::transform(marginals.begin(),marginals.end(),averageMarginal.begin(),marginals.begin(),std::plus<ValueType>());

			_postprocessMarginals(marginals.begin(),marginals.end());

			subSolver.IncreaseUnaryWeights(marginals.begin(),marginals.end());

			IndexType startNodeIndex=_core_order(0,_storage.size(modelIt->subModelId_));

			if (modelIt->subVariableId_!=startNodeIndex)
				subSolver.UpdateMarginals();
			    else subSolver.InitReverseMove();
		}
	}

	_FinalizeMove();
	_EvaluateIntegerBounds();
	_dualBound=_GetObjectiveValue();
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::_EstimateIntegerLabeling()
{
	for (IndexType i=0;i<_storage.numberOfSharedVariables();++i)
	{
		IndexType varId=_order(i);

		const typename Storage::SubVariableListType& varList=_storage.getSubVariableList(varId);
		_sumMarginal.assign(_storage.masterModel().numberOfLabels(varId),0.0);
		for(typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();modelIt!=varList.end();++modelIt)
		{
		 const_marginals_iterators_pair itpair=_subSolvers[modelIt->subModelId_]->GetMarginals(modelIt->subVariableId_);
		 _SumUpForwardMarginals(&_sumMarginal,itpair);
		}

		 typename PreviousFactorTable<GM>::const_iterator begin=_ftable.begin(varId,_moveDirection);
		 typename PreviousFactorTable<GM>::const_iterator end=_ftable.end(varId,_moveDirection);
		for (;begin!=end;++begin)
		{
		 if ((_factorProperties.getFunctionType(begin->factorId)==FunctionParameters<GM>::POTTS) && _parameters.fastComputations_)
		 {
			 _sumMarginal[_integerLabeling[begin->varId]]-=_factorProperties.getFunctionParameters(begin->factorId)[0];//instead of adding everywhere the same we just subtract the difference
		 }else
		 {
		 const typename GM::FactorType& pwfactor=_storage.masterModel()[begin->factorId];
		 IndexType localVarIndx = begin->localId;
		 LabelType fixedLabel=_integerLabeling[begin->varId];

			opengm::ViewFixVariablesFunction<GM> pencil(pwfactor,
					std::vector<opengm::PositionAndLabel<IndexType,LabelType> >(1,
							opengm::PositionAndLabel<IndexType,LabelType>(localVarIndx,
									fixedLabel)));

			for (LabelType j=0;j<_sumMarginal.size();++j)
				_sumMarginal[j]+=pencil(&j);
		 }
		}
		_EstimateIntegerLabel(varId,_sumMarginal);
	}
}

template <class SubSolver>
void TRWSPrototype<SubSolver>::_EvaluateIntegerBounds()
{
	_integerBound=_storage.masterModel().evaluate(_integerLabeling.begin());

	if (ACC::bop(_integerBound,_bestIntegerBound))
	{
		_bestIntegerLabeling=_integerLabeling;
		_bestIntegerBound=_integerBound;
	}

}

//================================= DecompositionStorage IMPLEMENTATION =================================================
template<class GM>
DecompositionStorage<GM>::DecompositionStorage(const GM& gm,StructureType structureType):
_gm(gm),
_structureType(structureType),
_subModels(),
_variableDecomposition(),
_var2FactorMap(gm)
{
	_InitSubModels();
}

template<class GM>
DecompositionStorage<GM>::~DecompositionStorage()
{
	for_each(_subModels.begin(),_subModels.end(),DeallocatePointer<SubModel>);
}

template<class GM>
void DecompositionStorage<GM>::_InitSubModels()
{
	Decomposition<GM>* pdecomposition;

	if (_structureType==GRIDSTRUCTURE)
		pdecomposition=new GridDecomposition<GM>(_gm);
	else
		pdecomposition=new MonotoneChainsDecomposition<GM>(_gm);

	try{
		pdecomposition->ComputeVariableDecomposition(&_variableDecomposition);
		size_t numberOfModels=pdecomposition->getNumberOfSubModels();
		_subModels.resize(numberOfModels);
		for (size_t modelId=0;modelId<numberOfModels;++modelId)
		{
			const typename SubModel::IndexList& varList=pdecomposition->getVariableList(modelId);
			typename SubModel::IndexList numOfSubModelsPerVar(varList.size());
			// Initialize numOfSubModelsPerVar
			for (size_t varIndx=0;varIndx<varList.size();++varIndx)
				numOfSubModelsPerVar[varIndx]=_variableDecomposition[varList[varIndx]].size();

			_subModels[modelId]= new SubModel(_gm,_var2FactorMap,varList,pdecomposition->getFactorList(modelId),numOfSubModelsPerVar);
		};
	}catch(std::runtime_error& err)
	{
		delete pdecomposition;
		throw err;
	}
};

#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void DecompositionStorage<GM>::PrintTestData(std::ostream& fout)const
{
	fout << "_variableDecomposition: "<<std::endl;
	for (size_t variableId=0;variableId<_variableDecomposition.size();++variableId)
	{
		std::for_each(_variableDecomposition[variableId].begin(),_variableDecomposition[variableId].end(),printSubVariable<typename MonotoneChainsDecomposition<GM>::SubVariable>(fout));
		fout << std::endl;
	}
}

template<class GM>
void DecompositionStorage<GM>::PrintVariableDecompositionConsistency(std::ostream& fout)const
{
	fout << "Variable decomposition consistency:" <<std::endl;
	for (size_t varId=0;varId<_gm.numberOfVariables();++varId)
	{
		fout << varId<<": ";
		const SubVariableListType& varList=_variableDecomposition[varId];
		typename SubVariableListType::const_iterator modelIt=varList.begin();
		std::vector<ValueType> sum(_gm.numberOfLabels(varId),0.0);
		while (modelIt!=varList.end())
		{
			const SubModel& subModel=*_subModels[modelIt->subModelId_];
			std::transform(subModel.unaryFactors(modelIt->subVariableId_).begin(),subModel.unaryFactors(modelIt->subVariableId_).end(),
			  			sum.begin(),sum.begin(),std::plus<ValueType>());
			++modelIt;
		}
		std::vector<ValueType> originalFactor(_gm.numberOfLabels(varId),0.0);
		_gm[varId].copyValues(originalFactor.begin());

		std::transform(sum.begin(),sum.end(),originalFactor.begin(),sum.begin(),std::minus<ValueType>());
		fout << std::accumulate(sum.begin(),sum.end(),(ValueType)0.0)<<std::endl;
	}

}
#endif
//================================= MaxSumTRWS IMPLEMENTATION =================================================

template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::_InitMove()
{
	parent::_moveDirection=SubModel::Direct;
	std::for_each(parent::_subSolvers.begin(), parent::_subSolvers.end(), std::mem_fun_t<void,SubSolver>(&SubSolver::InitMove));//!< calling ->InitMove() for each element of the container
}

template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::_postprocessMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end)
{
	transform_inplace(begin,end,std::bind1st(std::multiplies<ValueType>(),-1.0));
}

template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::_SumUpForwardMarginals(std::vector<ValueType>* pout,const_marginals_iterators_pair itpair)
{
	std::transform(itpair.first,itpair.second,pout->begin(),pout->begin(),std::plus<ValueType>());
}

template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::_EstimateTRWSBound()
{
	if (_canonicalNormalization) return;
	std::vector<ValueType> bounds(parent::_storage.numberOfModels());
	for (size_t i=0;i<bounds.size();++i)
		bounds[i]=parent::_subSolvers[i]->GetObjectiveValue();

	ValueType min=*std::min_element(bounds.begin(),bounds.end());
	ValueType max=*std::max_element(bounds.begin(),bounds.end());
	ValueType eps; ACC::iop(max-min,min-max,eps);
	ACC::iop(min,max,_pseudoBoundValue);
#ifdef TRWS_DEBUG_OUTPUT
	parent::_fout <<"min="<<min<<", max="<<max<<", eps="<<eps<<", pseudo bound="<<bounds.size()*_pseudoBoundValue<<std::endl;
#endif
}


template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::_normalizeMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end,SubSolver* subSolver)
{
	if (!_canonicalNormalization) return;
	ValueType maxVal=*std::max_element(begin,end,ACC::template bop<ValueType>);
	transform_inplace(begin,end,std::bind2nd(std::plus<ValueType>(),-maxVal));
}

template<class GM,class ACC>
void MaxSumTRWS<GM,ACC>::getTreeAgreement(std::vector<bool>& out)
{
	out.assign(parent::_storage.masterModel().numberOfVariables(),true);
	for (size_t varId=0;varId<parent::_storage.masterModel().numberOfVariables();++varId)
	{
		const typename Storage::SubVariableListType& varList=parent::_storage.getSubVariableList(varId);
		size_t label;
		for(typename Storage::SubVariableListType::const_iterator modelIt=varList.begin()
														;modelIt!=varList.end();++modelIt)
		{
			size_t check_label=parent::_subSolvers[modelIt->subModelId_]->arg()[modelIt->subVariableId_];
			if (modelIt==varList.begin())
			{
				label=check_label;
			}else if (check_label!=label)
			 {
				out[varId]=false;
				break;
			 }
		}

	}
}



template<class GM,class ACC>
bool MaxSumTRWS<GM,ACC>::CheckTreeAgreement(InferenceTermination* pterminationCode)
{
	  getTreeAgreement(_treeAgree);
	  size_t agree_count=count(_treeAgree.begin(),_treeAgree.end(),true);
#ifdef TRWS_DEBUG_OUTPUT
	  parent::_fout << "tree-agreement: " << agree_count <<" out of "<<_treeAgree.size() <<", ="<<100*(double)agree_count/_treeAgree.size()<<"%"<<std::endl;
#endif

	  if (_treeAgree.size()==agree_count)
	  {
#ifdef TRWS_DEBUG_OUTPUT
		  parent::_fout <<"Problem solved."<<std::endl;
#endif
		  *pterminationCode=opengm::CONVERGENCE;
		  return true;
	  }else
		  return false;
}


template<class GM,class ACC>
bool MaxSumTRWS<GM,ACC>::_CheckStoppingCondition(InferenceTermination* pterminationCode)
{
  if (CheckTreeAgreement(pterminationCode)) return true;

  return parent::_CheckStoppingCondition(pterminationCode);
}

//================================= SumProdTRWS IMPLEMENTATION =================================================
#ifdef TRWS_DEBUG_OUTPUT
template<class GM,class ACC>
void SumProdTRWS<GM,ACC>::PrintTestData(std::ostream& fout)const
{
	fout << "_smoothingValue:"<<_smoothingValue <<std::endl;
	parent::PrintTestData(fout);
}
#endif

template<class GM,class ACC>
void SumProdTRWS<GM,ACC>::_InitMove()//(ValueType smoothingValue)
{
	parent::_moveDirection=SubModel::Direct;
	std::for_each(parent::_subSolvers.begin(), parent::_subSolvers.end(), std::bind2nd(std::mem_fun(&SubSolver::InitMove),_smoothingValue));//!< calling ->InitMove() for each element of the container
}

template<class GM,class ACC>
void SumProdTRWS<GM,ACC>::_normalizeMarginals(typename std::vector<ValueType>::iterator begin,
											  typename std::vector<ValueType>::iterator end,SubSolver* subSolver)
{
	ValueType logPartition=subSolver->ComputeObjectiveValue();//!D not needed
	//normalizing marginals - subtracting log-partition function value/smoothing
	transform_inplace(begin,end,std::bind2nd(std::plus<ValueType>(),-logPartition/_smoothingValue));
}

template<class GM,class ACC>
void SumProdTRWS<GM,ACC>::_postprocessMarginals(typename std::vector<ValueType>::iterator begin,typename std::vector<ValueType>::iterator end)
{
	transform_inplace(begin,end,std::bind1st(std::multiplies<ValueType>(),-_smoothingValue));
}

template<class GM,class ACC>
void SumProdTRWS<GM,ACC>::_SumUpForwardMarginals(std::vector<ValueType>* pout,const_marginals_iterators_pair itpair)
{
	std::transform(pout->begin(),pout->end(),itpair.first,pout->begin(),plus2ndMul<ValueType>(_smoothingValue));
}

template<class GM,class ACC>
std::pair<typename SumProdTRWS<GM,ACC>::ValueType,typename SumProdTRWS<GM,ACC>::ValueType>
SumProdTRWS<GM,ACC>::GetMarginals(IndexType varId, OutputIteratorType begin)
{
  std::fill_n(begin,parent::_storage.numberOfLabels(varId),0.0);

  const typename Storage::SubVariableListType& varList=parent::_storage.getSubVariableList(varId);
  for(typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();modelIt!=varList.end();++modelIt)
  {
	  typename SubSolver::const_iterators_pair marginalsit=parent::_subSolvers[modelIt->subModelId_]->GetMarginals(modelIt->subVariableId_);
		std::vector<ValueType>& normMarginals=parent::_marginals[modelIt->subModelId_];
		normMarginals.resize(parent::_storage.numberOfLabels(varId));
	  //normalize
	  std::copy(marginalsit.first,marginalsit.second,normMarginals.begin());
	  _normalizeMarginals(normMarginals.begin(),normMarginals.end(),parent::_subSolvers[modelIt->subModelId_]);
	  ValueType mul; ACC::op(1.0,-1.0,mul);
	  transform_inplace(normMarginals.begin(),normMarginals.end(),mulAndExp<ValueType>(mul));
	  std::transform(normMarginals.begin(),normMarginals.end(),begin,begin,std::plus<ValueType>());
  }
  transform_inplace(begin,begin+parent::_storage.numberOfLabels(varId),std::bind1st(std::multiplies<ValueType>(),1.0/varList.size()));

  ValueType ell2Norm=0, ellInftyNorm=0;
  for (typename Storage::SubVariableListType::const_iterator modelIt=varList.begin();modelIt!=varList.end();++modelIt)
  {
	  std::vector<ValueType>& normMarginals=parent::_marginals[modelIt->subModelId_];
	  OutputIteratorType begin0=begin;
	  for (typename std::vector<ValueType>::const_iterator bm=normMarginals.begin(); bm!=normMarginals.end();++bm)
	  {
		  ValueType diff=(*bm-*begin0); ++begin0;
		  ell2Norm+=diff*diff;
		  ellInftyNorm=std::max((ValueType)fabs(diff),ellInftyNorm);
	  }
  }

  return std::make_pair(sqrt(ell2Norm),ellInftyNorm);
}

template<class GM,class ACC>
typename SumProdTRWS<GM,ACC>::ValueType
SumProdTRWS<GM,ACC>::GetMarginalsAndDerivativeMove()
{
	ValueType derivativeValue=0.0;
	//std::for_each(parent::_subSolvers.begin(), parent::_subSolvers.end(), std::(&SubSolver::MoveBackGetDerivative()));
	for (size_t i=0;i<parent::_subSolvers.size();++i)
		derivativeValue+=parent::_subSolvers[i]->MoveBackGetDerivative();

	parent::_moveDirection=SubModel::ReverseDirection(parent::_moveDirection);
	return derivativeValue;
}

};//DD

#endif /* ADSAL_H_ */
