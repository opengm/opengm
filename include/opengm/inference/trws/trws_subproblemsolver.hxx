#ifndef TRWS_SUBPROBLEMSOLVER_HXX_
#define TRWS_SUBPROBLEMSOLVER_HXX_
#include <iostream>
#include <list>
#include <algorithm>
#include <utility>
#include <functional>
#include <valarray>

#include <opengm/inference/trws/utilities2.hxx>
#include <opengm/functions/view_fix_variables_function.hxx>

#ifdef TRWS_DEBUG_OUTPUT
#include <opengm/inference/trws/output_debug_utils.hxx>
#endif

namespace opengm {
namespace trws_base{

#ifdef TRWS_DEBUG_OUTPUT
using OUT::operator <<;
#endif

template<class GM>
class SequenceStorage
{
public:
	//typedef GraphicalModel GM;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef std::vector<IndexType> IndexList;
	typedef std::vector<ValueType> UnaryFactor;
	typedef enum{Direct,Reverse} MoveDirection;
	typedef VariableToFactorMapping<GM> VariableToFactorMap;

	SequenceStorage(const GM& masterModel,const VariableToFactorMap& var2FactorMap,const IndexList& variableList, const IndexList& pwFactorList,const IndexList& numberOfTreesPerFactor);

	~SequenceStorage(){};
	IndexType size()const{return (IndexType)_directIndex.size();};//<! returns number of variables in the sequence

#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
#endif

	static MoveDirection ReverseDirection(MoveDirection dir);

	/*
	 * Serving functions:
	 */
	/*
	 * allocates the container (*pfactors) with sizes, corresponding to all unary factors of the associated graphoical model
	 */
	void AllocateUnaryFactors(std::vector<UnaryFactor>* pfactors);
	MoveDirection pwDirection(IndexType pwInd)const{assert(pwInd<_pwDirection.size()); return _pwDirection[pwInd];};
	IndexType pwForwardFactor(IndexType var)const{assert(var<_pwForwardIndex.size()); return _pwForwardIndex[var];}
	const GM& masterModel()const{return _masterModel;}
	/*
	 * unary factors access
	 */
	const UnaryFactor& unaryFactors(IndexType indx)const{assert(indx<_unaryFactors.size()); return _unaryFactors[indx];}//!>const access
	typename UnaryFactor::iterator ufBegin(IndexType indx){assert(indx<_unaryFactors.size()); return _unaryFactors[indx].begin();}//!>non-const access TODO: make a pair of iterators from a single function call
	typename UnaryFactor::iterator ufEnd  (IndexType indx){assert(indx<_unaryFactors.size()); return _unaryFactors[indx].end()  ;}//!>non-const access
	IndexType varIndex(IndexType var)const{assert(var<_directIndex.size()); return _directIndex[var];};

	template<class ITERATOR>
	ValueType evaluate(ITERATOR labeling);
private:
	void _ConsistencyCheck();
	void _Reset(const IndexList& numOfSequencesPerFactor);//TODO: set weights from a vector
	void _Reset(IndexType var,IndexType numOfSequences);//TODO: set weights from a vector

	const GM& _masterModel;
	/** var - in local coordinates (of the subProblem), var can be transformed from varIndex by _variable() function */
	IndexList _directIndex;
	IndexList _pwForwardIndex;
	std::vector<UnaryFactor> _unaryFactors;
	std::vector<MoveDirection> _pwDirection;
	const VariableToFactorMap& _var2FactorMap;
};

//===== class FunctionParameters ========================================

template<class GM>
class FunctionParameters
{
public:
	typedef enum {GENERAL,POTTS} FunctionType;
	typedef typename GM::ValueType ValueType;
//	typedef std::valarray<ValueType> ParameterStorageType;
	typedef std::vector<ValueType> ParameterStorageType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;

	FunctionParameters(const GM& gm);
	FunctionType getFunctionType(IndexType factorId)const
	{
		OPENGM_ASSERT(factorId<_factorTypes.size());
		return _factorTypes[factorId];
	};
	const ParameterStorageType& getFunctionParameters(IndexType factorId)const
	{
	//	_checkConsistency();
		OPENGM_ASSERT(factorId < _parameters.size());
		return _parameters[factorId];
	}
#ifdef TRWS_DEBUG_OUTPUT
	void PrintStatusData(std::ostream& fout);
#endif
private:
	void _checkConsistency() const;
	void _getPottsParameters(const typename GM::FactorType& factor,ParameterStorageType* pstorage)const;
	const GM& _gm;
	std::vector<ParameterStorageType> _parameters;
	std::vector<FunctionType> _factorTypes;
};

template<class GM>
FunctionParameters<GM>::FunctionParameters(const GM& gm)
: _gm(gm),_parameters(_gm.numberOfFactors()),_factorTypes(_gm.numberOfFactors())
{
	for (IndexType i=0;i<_gm.numberOfFactors();++i)
	{
		const typename GM::FactorType& f=_gm[i];

		if ((f.numberOfVariables()==2) && f.isPotts())
		{
			_factorTypes[i]=POTTS;
			_getPottsParameters(f,&_parameters[i]);
		}else	_factorTypes[i]=GENERAL;
	}

//	_checkConsistency();
}

template<class GM>
void  FunctionParameters<GM>::_checkConsistency()const
{
	OPENGM_ASSERT(_parameters.size()==_gm.numberOfFactors());
	OPENGM_ASSERT(_factorTypes.size()==_gm.numberOfFactors());
	for (size_t i=0;i<_parameters.size();++i)
		if (_factorTypes[i]==POTTS)
		{
			OPENGM_ASSERT(_parameters[i].size()==2);
		}
}

template<class GM>
void FunctionParameters<GM>::_getPottsParameters(const typename GM::FactorType& f,ParameterStorageType* pstorage)const
{
//	pstorage->resize(2,0.0);
	pstorage->assign(2,0.0);
	LabelType v00[]={0,0};
	LabelType v01[]={0,1};
	LabelType v10[]={1,0};
	if ((f.numberOfLabels(0)>0) && (f.numberOfLabels(1)>0))
		(*pstorage)[1]=f(&v00[0]);
	if (f.numberOfLabels(0)>1)
		(*pstorage)[0]=f(&v10[0])-f(&v00[0]);
	else if (f.numberOfLabels(1)>1)//BSD: bug fixed.
		(*pstorage)[0]=f(&v01[0])-f(&v00[0]);
}

#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void FunctionParameters<GM>:: PrintStatusData(std::ostream& fout)
{
	size_t numPotts=0;
	for (size_t i=0;i<_parameters.size();++i) numPotts+= (_factorTypes[i]==POTTS ? 1 : 0) ;
	fout << "Total number of factors:" <<_factorTypes.size()<<std::endl;
	fout << "Number of POTTS p/w factors:" << numPotts <<std::endl;
}
#endif

//===== class Dynamic Programming ========================================

template<class GM,class ACC,class InputIterator>
class DynamicProgramming
{
public:
typedef GM GMType;
typedef ACC ACCType;
typedef typename GM::ValueType ValueType;
typedef typename GM::IndexType IndexType;
typedef typename GM::LabelType LabelType;

typedef InputIterator InputIteratorType;
typedef SequenceStorage<GM> Storage;
typedef typename Storage::IndexList IndexList;
typedef typename Storage::UnaryFactor UnaryFactor;
typedef typename Storage::MoveDirection MoveDirection;
typedef std::vector<IndexList> IndexTable;
typedef FunctionParameters<GM> FactorProperties;
typedef typename UnaryFactor::const_iterator ConstIterator;
typedef typename GM::FactorType Factor;
typedef std::pair<typename UnaryFactor::const_iterator,typename UnaryFactor::const_iterator> const_iterators_pair;

public:
static const IndexType NaN;//=std::numeric_limits<IndexType>::max();

DynamicProgramming(Storage& storage,const FactorProperties& factorProperties,bool fastComputations=true);//:_storage(storage){};
//private:
virtual ~DynamicProgramming(){};
//public:
/**
 * Inference: usage:
 * InitMove(rho);//once initialize smoothing value
 *
 * then
 *
 * Move();/MoveBack();
 *parent::_moveDirection
 * or
 *
 * PushBack();....PushBack();FinalizeMove();
 */
void InitMove(){_InitMove(1.0,Storage::Direct);};
void InitMove(MoveDirection movedirection){_InitMove(1.0,movedirection);};
virtual void InitReverseMove(){_InitMove(_rho,Storage::ReverseDirection(_moveDirection));};//!>initializes move, which is reverse to the current one//TODO: remove virtual ?
virtual void Move();//performs forward move//TODO: remove virtual ?
virtual void PushBack();//performs a single step of the move and sums up corresponding fw-bk marginals//TODO: remove virtual ?
virtual void MoveBack();//performs size() steps with PushBack();//TODO: remove virtual ?
/**
 * Returns NON-normalized marginals ...(logSumProd or maxSum marginals)
 */
const_iterators_pair GetMarginals()const{return std::make_pair(_marginals[_currentUnaryIndex].begin(),_marginals[_currentUnaryIndex].end());};
const_iterators_pair GetMarginals(IndexType indx)const{assert(indx<_marginals.size()); return std::make_pair(_marginals[indx].begin(),_marginals[indx].end());};

ValueType  GetObjectiveValue()const{return _objectiveValue;};
/*
 * Returns value of the objective partition function, corresponding to the marginals returned by GetMarginals()
 */
virtual ValueType ComputeObjectiveValue()=0;//{return ACC::neutral<ValueType>();}
/*
 * increases weights of the current unary factor and corresponding temporary array _currentUnaryFactor
 * the end-begin should be defined and equal to the number of labels in the current unary factor
 */

virtual void IncreaseUnaryWeights(InputIteratorType begin,InputIteratorType end);
/*
 * call it if you have performed operations of the move by calling Push()/PushBack() and want to have _logPartition computed correctly
 */
virtual void FinalizeMove();
/**
 * Returns number of labels in the current node
 */
LabelType numOfLabels()const{const_iterators_pair p=GetMarginals(); return p.second-p.first;}
virtual void UpdateMarginals();//!> updates marginals in the current node so, that they correspond to the forward (backward) accumulated probabilities of labels

virtual IndexType getNextPWId()const;//!> returns an external (_gm[.]) pairwise index, which follows the current variable. For the last variable this::NaN is returned
virtual IndexType getPrevPWId()const;//!> returns an external (_gm[.]) pairwise index, which is in front of the current variable. For the first variable this::NaN is returned

MoveDirection  getMoveDirection()const{return _moveDirection;}
IndexType size()const{return (IndexType)_storage.size();}
template<class ITERATOR>
ValueType evaluate(ITERATOR labeling){return _storage.evaluate(labeling);}
/**
 * Tests
 */
#ifdef TRWS_DEBUG_OUTPUT
virtual void PrintTestData(std::ostream& fout)const;
#endif

void SetFastComputation(bool fc){_fastComputation=fc;}

protected:

void _PottsUnaryTransform(LabelType newSize,const typename FactorProperties::ParameterStorageType& params);

void _InitReverseMoveBack(){_core_InitMoves(_rho,Storage::ReverseDirection(_moveDirection));};//!>initializes move, which is reverse to the current one
void _InitMove(ValueType rho,MoveDirection movedirection);
virtual void _Push();//performs a single step of the move
void _core_InitMoves(ValueType rho,MoveDirection movedirection);
void _PushMessagesToFactor();//updates _currentPWFactor+=marginals
void _ClearMessages(UnaryFactor* pbuffer=0);//makes 0 message in each p/w pencil; updates _currentPWFactor and _marginals(begin0+1)
virtual void _makeLocalCopyOfPWFactor(LabelType trgsize);//makes a local copy of a p/w factor taking into account the processing order
void _SumUpBufferToMarginals();
virtual void _BackUpForwardMarginals(){};
virtual void _InitCurrentUnaryBuffer(IndexType index);

IndexType _core_next(IndexType begin,MoveDirection dir)const;
IndexType _next(IndexType begin)const;
IndexType _previous(IndexType begin)const;
IndexType _nextPWIndex()const;

bool _fastComputation;
Storage& _storage;
const FactorProperties& _factorProperties;

std::vector<UnaryFactor> _marginals;

ValueType 				 _objectiveValue;
ValueType 				 _rho; //current smoothing constant
MoveDirection			 _moveDirection;
bool 					 _bInitializationNeeded;

//------processing data for a current step
UnaryFactor _currentPWFactor;
UnaryFactor _currentUnaryFactor;
IndexType      _currentUnaryIndex;
//------Calculation optimizations
mutable UnaryFactor _unaryTemp;
mutable Pseudo2DArray<ValueType> _spst;
};

//typedef DynamicProgramming MaxSumSolver;
template<class GM,class ACC,class InputIterator>
class MaxSumSolver : public DynamicProgramming<GM,ACC,InputIterator>
{
public:
	typedef DynamicProgramming<GM,ACC,InputIterator> parent;
	typedef typename parent::ValueType ValueType;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;
	typedef typename parent::InputIteratorType InputIteratorType;
	typedef std::vector<LabelType> LabelingType;
	typedef typename parent::UnaryFactor UnaryFactor;
	typedef typename parent::Factor Factor;
	typedef typename parent::FactorProperties FactorProperties;

	//MaxSumSolver(Storage& storage):parent(storage){};
	MaxSumSolver(typename parent::Storage& storage,const FactorProperties& factorProperties,bool fastComputations=true)
				 :parent(storage,factorProperties,fastComputations),
			 	 _labeling(parent::size(),parent::NaN)
	//		 	 ,_factorParameters(2,0.0)
	{};

#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const
	{
		parent::PrintTestData(fout);
		fout <<	"_labeling: "<<_labeling<<std::endl;
	}
#endif

	ValueType ComputeObjectiveValue();
	const LabelingType& arg(){return _labeling;}

	void FinalizeMove();

protected:
	void _Push();
	void _SumUpBackwardEdges(UnaryFactor* u, LabelType fixedLabel)const;
	void _EstimateOptimalLabeling();
	LabelingType			 _labeling;
	mutable UnaryFactor _marginalsTemp;
//	mutable typename FactorProperties::ParameterStorageType _factorParameters;
};

template<class GM,class ACC,class InputIterator>
void MaxSumSolver<GM,ACC,InputIterator>::_EstimateOptimalLabeling()
{
 OPENGM_ASSERT((parent::_currentUnaryIndex==0)||(parent::_currentUnaryIndex==parent::size()-1));
 OPENGM_ASSERT(_labeling[parent::_currentUnaryIndex]<parent::_marginals[parent::_currentUnaryIndex].size());
 //Backup _currentUnaryIndex
 IndexType bk_currentUnaryIndex=parent::_currentUnaryIndex;
 //... and _MoveDirection
 typename parent::MoveDirection bk_moveDirection=parent::_moveDirection;
 parent::_moveDirection=parent::Storage::ReverseDirection(parent::_moveDirection);

 //move to the end and compute the sum. Use View function of the GM
 LabelType optLabel=_labeling[parent::_currentUnaryIndex];

 for (IndexType i=1;i<parent::size();++i)
 {
	  parent::_currentUnaryIndex=parent::_next(parent::_currentUnaryIndex);
	  _marginalsTemp=parent::_marginals[parent::_currentUnaryIndex];
	  _SumUpBackwardEdges(&_marginalsTemp,optLabel);

	  _labeling[parent::_currentUnaryIndex]=optLabel=std::max_element(_marginalsTemp.begin(),_marginalsTemp.end(),
													   ACC::template ibop<ValueType>)-_marginalsTemp.begin();
 }

 //restore the _currentUnaryIndex and _MoveDirection
 parent::_moveDirection=bk_moveDirection;
 parent::_currentUnaryIndex=bk_currentUnaryIndex;
}

template<class GM,class ACC,class InputIterator>
typename MaxSumSolver<GM,ACC,InputIterator>::ValueType
MaxSumSolver<GM,ACC,InputIterator>::ComputeObjectiveValue()
{
	_labeling[parent::_currentUnaryIndex]=std::max_element(parent::_marginals[parent::_currentUnaryIndex].begin(),
			  parent::_marginals[parent::_currentUnaryIndex].end(),ACC::template ibop<ValueType>)
			  -parent::_marginals[parent::_currentUnaryIndex].begin();
	return parent::_marginals[parent::_currentUnaryIndex][_labeling[parent::_currentUnaryIndex]];
}

template<class GM,class ACC,class InputIterator>
void MaxSumSolver<GM,ACC,InputIterator>::FinalizeMove()
{
	parent::FinalizeMove();
	_EstimateOptimalLabeling();
};

template <class T,class ACC> struct compToValue : std::unary_function <T,T> {
	compToValue(T val):_val(val){};
  T operator() (T x) const
    {return (ACC::template bop<T>(x,_val) ? x : _val);}
private:
  T _val;
};

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_PottsUnaryTransform(LabelType newSize,const typename FactorProperties::ParameterStorageType& params)
{
	OPENGM_ASSERT(params.size()==2);
	UnaryFactor* puf=&(_currentUnaryFactor);

//	if (newSize< puf->size())
//		puf->resize(newSize);//Bug!

	typename UnaryFactor::iterator bestValIt=std::max_element(puf->begin(),puf->end(),ACC::template ibop<ValueType>);
	ValueType bestVal=*bestValIt;
	ValueType secondBestVal=bestVal;
//if (puf->size()>1){
	if (ACC::bop(params[0],static_cast<ValueType>(0.0)))//!> if anti-Potts model
	{
		*bestValIt=ACC::template neutral<ValueType>();
		secondBestVal=*std::max_element(puf->begin(),puf->end(),ACC::template ibop<ValueType>);
		*bestValIt=bestVal;
	}
//}else{std::cout << "1: puf->size()="<<puf->size()<<std::endl;}

	transform_inplace(puf->begin(),puf->end(),compToValue<ValueType,ACC>(bestVal+params[0]));

//if (puf->size()>1){
	if (ACC::bop(params[0],static_cast<ValueType>(0.0)))//!> if anti-Potts model
		ACC::op(secondBestVal+params[0],bestVal,*bestValIt);
//}else{std::cout << "2: puf->size()="<<puf->size()<<std::endl;}

	if (params[1]!=0.0)
		transform_inplace(puf->begin(),puf->end(),std::bind1st(std::plus<ValueType>(),params[1]));

	if (newSize< puf->size())
		puf->resize(newSize);//BSD: Bug fixed?
	else if (newSize > puf->size())
	{
		puf->resize(newSize,params[0]+params[1]+bestVal);
//		std::cout <<"puf.size()="<<puf->size()<<", bestVal="<<bestVal<<", params[0]="<<params[0]<<", params[1]="<<params[1]
//				<<", (*puf)[1]="<<(*puf)[1]<<std::endl;
	}

}

template<class GM,class ACC,class InputIterator>
void MaxSumSolver<GM,ACC,InputIterator>::_Push()
{
 IndexType factorId=parent::_storage.pwForwardFactor(parent::_nextPWIndex());
 if ((parent::_factorProperties.getFunctionType(factorId)==FunctionParameters<GM>::POTTS) && parent::_fastComputation)
 {
	 parent::_currentUnaryIndex=parent::_next(parent::_currentUnaryIndex);
	 LabelType newSize=parent::_storage.unaryFactors(parent::_currentUnaryIndex).size();
	 parent::_PottsUnaryTransform(newSize,parent::_factorProperties.getFunctionParameters(factorId));
	 std::transform(parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.end(),
			       parent::_storage.unaryFactors(parent::_currentUnaryIndex).begin(),
			       parent::_currentUnaryFactor.begin(),plus2ndMul<ValueType>(1.0/parent::_rho));
 }else
	 parent::_Push();
}


//===== class SumProdSequenceSolver ========================================


template<class GM,class ACC,class InputIterator>
class SumProdSolver : public DynamicProgramming<GM,ACC,InputIterator>
{
public:
typedef DynamicProgramming<GM,ACC,InputIterator> parent;
typedef typename parent::ValueType ValueType;
typedef typename parent::IndexType IndexType;
typedef typename parent::LabelType LabelType;
typedef typename parent::InputIteratorType InputIteratorType;
typedef typename parent::const_iterators_pair const_iterators_pair;
typedef typename parent::Storage Storage;
typedef typename parent::MoveDirection MoveDirection;
typedef typename parent::UnaryFactor UnaryFactor;
typedef typename parent::FactorProperties FactorProperties;


SumProdSolver(Storage& storage,const FactorProperties& factorProperties,bool fastComputations=true)
:parent(storage,factorProperties,fastComputations),_averagingFlag(false){ACC::op(1.0,-1.0,_mul);};
void InitMove(ValueType rho){parent::_InitMove(rho,Storage::Direct);};
void InitMove(ValueType rho,MoveDirection movedirection){parent::_InitMove(rho,movedirection);};

ValueType ComputeObjectiveValue();
ValueType MoveBackGetDerivative();//!>makes MoveBack and returns derivative w.r.t. _smoothingValue
ValueType getDerivative()const{return _derivativeValue;}
protected:
void _Push();//performs a single step of the move
void _ExponentiatePWFactor();//updates _currentPWFactor - exponentiation in place
void _PushMessagesToVariable();//sums up p/w pencils, updates _currentUnaryFactor=sum  and _marginals(begin0+1) += log(_currentUnaryFactor) + _unaryFactor
void _PushAndAverage();//additionally to _Push performs estimation of the PW average potentials
void _UpdatePWAverage();
ValueType _getMarginalsLogNormalizer()const{return parent::GetObjectiveValue()/parent::_rho;}//!> subtract it if you want to get normalized log-marginals from non-normalized ones
ValueType _GetAveragedUnaryFactors();
void _makeLocalCopyOfPWFactor(LabelType trgsize);//makes a local copy of a p/w factor taking into account the processing order
void _InitCurrentUnaryBuffer(IndexType index);

ValueType _mul;
bool _averagingFlag;
/*
 * optimization of computations
 */

UnaryFactor _unaryBuffer;
UnaryFactor _copyPWfactor;
ValueType _derivativeValue;
};


//=======================SequenceStorage implementation ===========================================
#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void SequenceStorage<GM>::PrintTestData(std::ostream& fout)const
{
fout << "_directIndex:" <<_directIndex;
fout << "_pwForwardIndex:" <<_pwForwardIndex;
fout << "_unaryFactors:" <<std::endl<<_unaryFactors;
fout << "_pwDirection:" << _pwDirection;
};
#endif

template<class GM>
SequenceStorage<GM>::SequenceStorage(const GM& masterModel,const VariableToFactorMap& var2FactorMap,
		const IndexList& variableList,
		const IndexList& pwFactorList,
		const IndexList& numOfSequencesPerFactor)//TODO: exchange to the vector of initial values
:_masterModel(masterModel),
 _directIndex(variableList),
 _pwForwardIndex(pwFactorList),
 _pwDirection(pwFactorList.size())
 ,_var2FactorMap(var2FactorMap)
{
	_ConsistencyCheck();
	AllocateUnaryFactors(&_unaryFactors);
	_Reset(numOfSequencesPerFactor);//TODO: set weights from a vector
}

template<class GM>
void SequenceStorage<GM>::_ConsistencyCheck()
{
	exception_check((_directIndex.size()-1)==_pwForwardIndex.size(),"DynamicProgramming::_ConsistencyCheck(): (_directIndex.size()-1)!=_pwForwardIndex.size()");

	 LabelType v[2];
	 for (IndexType i=0;i<size()-1;++i)
	 {
	  exception_check(_masterModel[pwForwardFactor(i)].numberOfVariables()==2,"DynamicProgramming::_ConsistencyCheck():factor.numberOfVariables()!=2");
	  _masterModel[pwForwardFactor(i)].variableIndices(&v[0]);

	  if (v[0]==varIndex(i))
	  {
		  exception_check(v[1]==varIndex(i+1),"DynamicProgramming::_ConsistencyCheck(): v[1]!=varIndex(i+1)");
		  _pwDirection[i]=Direct;
	  }
	  else if (v[0]==varIndex(i+1))
	  {
		  exception_check(v[1]==varIndex(i),"DynamicProgramming::_ConsistencyCheck(): v[1]!=varIndex(i)");
		  _pwDirection[i]=Reverse;
	  }
	  else
		  throw std::runtime_error("DynamicProgramming::_ConsistencyCheck(): pairwise factor does not correspond to unaries!");
	 }
}

template<class GM>
void SequenceStorage<GM>::_Reset(const IndexList& numOfSequencesPerFactor)
{
	for (IndexType var=0;var<size();++var)
		_Reset(var,numOfSequencesPerFactor[var]);
};

template<class GM>
void SequenceStorage<GM>::_Reset(IndexType var,IndexType numOfSequences)
{
	assert(var<size());
	UnaryFactor& uf=_unaryFactors[var];
	_masterModel[_var2FactorMap(varIndex(var))].copyValues(uf.begin());
	transform_inplace(uf.begin(),uf.end(),std::bind2nd(std::multiplies<ValueType>(),1.0/numOfSequences));

};

template<class GM>
void SequenceStorage<GM>::AllocateUnaryFactors(std::vector<UnaryFactor>* pfactors)
{
 pfactors->resize(size());
 for (size_t i=0;i<pfactors->size();++i)
		 (*pfactors)[i].assign(_masterModel[_var2FactorMap(varIndex(i))].size(),0.0);
};

template<class GM>
typename SequenceStorage<GM>::MoveDirection SequenceStorage<GM>::ReverseDirection(MoveDirection dir)
{
	if (dir==Direct)
		return Reverse;
	else
		return Direct;
}

template<class GM>
template<class ITERATOR>
typename SequenceStorage<GM>::ValueType
SequenceStorage<GM>::evaluate(ITERATOR labeling)
{
	ValueType value=0.0;
	for (size_t i=0;i<size();++i)
	{
		value+=_unaryFactors[i][*labeling];
		if (i<size()-1)
		{
		 if (pwDirection(i)==Direct)
		  value+=_masterModel[_pwForwardIndex[i]](labeling);
		 else
		 {
		  std::valarray<LabelType> ind(2);
		  ind[0]=*(labeling+1); ind[1]=*labeling;
		  value+= _masterModel[_pwForwardIndex[i]](labeling);
		 }
		}
		++labeling;
	}
	return value;
}

//========================DynamicProgramming  Implementation =============================================
template<class GM,class ACC,class InputIterator>
const typename DynamicProgramming<GM,ACC,InputIterator>::IndexType DynamicProgramming<GM,ACC,InputIterator>::NaN=std::numeric_limits<IndexType>::max();

template<class GM,class ACC,class InputIterator>
DynamicProgramming<GM,ACC,InputIterator>::DynamicProgramming(Storage& storage,const FactorProperties& factorProperties,bool fastComputation)
:_fastComputation(fastComputation),
 _storage(storage),
 _factorProperties(factorProperties),
 _objectiveValue(0.0),
 _rho(1.0),
 _moveDirection(Storage::Direct),
 _bInitializationNeeded(true),
 _currentPWFactor(0),
 _currentUnaryFactor(0),
 //_currentUnaryIndex(std::numeric_limits<size_t>::max())
 _currentUnaryIndex(NaN)
 {
	_storage.AllocateUnaryFactors(&_marginals);
};

#ifdef TRWS_DEBUG_OUTPUT
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::PrintTestData(std::ostream& fout)const
{
fout << "_marginals:" <<std::endl<<_marginals;
fout << "_objectiveValue="<<_objectiveValue<<std::endl;
fout << "_rho="<<_rho<<std::endl;
fout <<	"_moveDirection="<< _moveDirection<<std::endl;
fout << "_currentPWFactor="<<_currentPWFactor;
fout << "_currentUnaryFactor="<<_currentUnaryFactor;
fout << "_currentUnaryIndex=" <<_currentUnaryIndex<<std::endl;
};
#endif

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::_core_next(IndexType begin,MoveDirection dir)const
{
	if (dir==Storage::Direct)
	{
	 assert(begin<_storage.size()-1);
	 return ++begin;
	}
	else
	{
	 assert((begin>0) && (begin<_storage.size()));
	 return --begin;
	}
}

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::_next(IndexType begin)const
{
	return _core_next(begin,_moveDirection);
}

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::_previous(IndexType begin)const
{
 if (_moveDirection==Storage::Direct)
	 return _core_next(begin,Storage::Reverse);
 else
	 return _core_next(begin,Storage::Direct);
}

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::_nextPWIndex()const
{
  if (_moveDirection==Storage::Direct)
	  return _currentUnaryIndex;
  else
	  return _currentUnaryIndex-1;
}

//makes a local copy of a p/w factor taking into account the processing order
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_makeLocalCopyOfPWFactor(LabelType trgsize)
{
const Factor& f=_storage.masterModel()[_storage.pwForwardFactor(_nextPWIndex())];
_currentPWFactor.resize(f.size());
if (    ((_moveDirection==Storage::Direct) && (_storage.pwDirection(_nextPWIndex())==Storage::Direct)) ||
		((_moveDirection==Storage::Reverse) && (_storage.pwDirection(_nextPWIndex())==Storage::Reverse)) )
	f.copyValues(_currentPWFactor.begin());
else
	f.copyValuesSwitchedOrder(_currentPWFactor.begin());
}

//move unaries to fwMessages
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_PushMessagesToFactor()
{
	LabelType trgsize=_storage.unaryFactors(_next(_currentUnaryIndex)).size();//check asserts of _next first

	//coping pw factor to the temporary storage
	_makeLocalCopyOfPWFactor(trgsize);
	assert(_currentPWFactor.size()==(_currentUnaryFactor.size()*trgsize));

	if (_rho!=1.0)  std::transform(_currentPWFactor.begin(),_currentPWFactor.end(),_currentPWFactor.begin(),std::bind2nd(std::multiplies<ValueType>(),1.0/_rho));

	_spst.resize(_currentUnaryFactor.size(),trgsize);

	//increase each pencil of the p/w factor to the value of the marginal
	for (LabelType i=0;i<_currentUnaryFactor.size();++i)
		transform_inplace(_spst.beginSrcNC(&_currentPWFactor[0],i),_spst.endSrcNC(&_currentPWFactor[0],i),std::bind2nd(std::plus<ValueType>(),_currentUnaryFactor[i]));
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_InitCurrentUnaryBuffer(IndexType index)
{
	assert(index < _storage.size());
	_currentUnaryIndex=index;
	_currentUnaryFactor.resize(_storage.unaryFactors(_currentUnaryIndex).size());
	std::copy(_storage.unaryFactors(_currentUnaryIndex).begin(),_storage.unaryFactors(_currentUnaryIndex).end(),_currentUnaryFactor.begin());
}

template<class T,class Iterator,class Comp>
 T _MaxNormalize_inplace(Iterator begin, Iterator end, T init,Comp comp)
 {
 	T max=*std::max_element(begin,end,comp);
 	transform_inplace(begin,end,std::bind2nd(std::minus<T>(),max));
 	return init+max;
 }

//clear bkMessages
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_ClearMessages(UnaryFactor* pbuffer)
{
	LabelType srcsize=_storage.unaryFactors(_previous(_currentUnaryIndex)).size();//check asserts of _previous first

	_spst.resize(srcsize,_currentUnaryFactor.size());

	if (pbuffer==0)
	{
	 for (LabelType i=0;i<_currentUnaryFactor.size();++i)
		_currentUnaryFactor[i]+=_MaxNormalize_inplace(_spst.beginTrgNC(&_currentPWFactor[0],i),_spst.endTrgNC(&_currentPWFactor[0],i),(ValueType)0.0,ACC::template ibop<ValueType>);
	}
	else
	{
		pbuffer->resize(_currentUnaryFactor.size());
		for (LabelType i=0;i<_currentUnaryFactor.size();++i)
		  _currentUnaryFactor[i]+=(*pbuffer)[i]=_MaxNormalize_inplace(_spst.beginTrgNC(&_currentPWFactor[0],i),_spst.endTrgNC(&_currentPWFactor[0],i),(ValueType)0.0,ACC::template ibop<ValueType>);
	}
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_Push()
{
	//move unaries to fwMessages
	_PushMessagesToFactor();//updates _currentPWFactor[pencil i]+=_currentUnaryFactor[i]
	_InitCurrentUnaryBuffer(_next(_currentUnaryIndex));
	//clear bkMessages
	_ClearMessages();//updates _currentPWFactor, that each pencil contains 0 and _currUnaryFactor = unaryFactor/rho + pencil normalization
	_BackUpForwardMarginals();//TODO: check me!
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::UpdateMarginals()
{
	std::copy(_currentUnaryFactor.begin(),_currentUnaryFactor.end(),_marginals[_currentUnaryIndex].begin());
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_SumUpBufferToMarginals()
{
	UnaryFactor& marginals=_marginals[_currentUnaryIndex];
	std::transform(_currentUnaryFactor.begin(),_currentUnaryFactor.end(),marginals.begin(),marginals.begin(),std::plus<ValueType>());
	std::transform(marginals.begin(),marginals.end(),_storage.unaryFactors(_currentUnaryIndex).begin(),marginals.begin(),plus2ndMul<ValueType>(-1.0/_rho));
}

//performs forward move
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::Move()
{
  if (_bInitializationNeeded)
  {
	InitReverseMove();
	_bInitializationNeeded=false;
  }
	//push
 for (IndexType i=0;i<_storage.size()-1;++i)
 {
	 _Push();
	 UpdateMarginals();
 }

 //_NormalizeMarginals();
 FinalizeMove();
}

/**
 * performs a single step of the move and sums up corresponding fw-bk marginals
 */
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::PushBack()
{
	if (_bInitializationNeeded)
	{
		_InitReverseMoveBack();
	}

	_Push();
	_SumUpBufferToMarginals();
}

/**
 * performs size() steps with PushBack();
 */
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::MoveBack()
{
//push
 for (IndexType i=0;i<_storage.size()-1;++i)
	 PushBack();

 FinalizeMove();
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_core_InitMoves(ValueType rho,MoveDirection movedirection)
{
	_rho=rho;
	_moveDirection=movedirection;

	if (_moveDirection==Storage::Direct)
		_InitCurrentUnaryBuffer(0);
	else
		_InitCurrentUnaryBuffer(_storage.size()-1);

	_bInitializationNeeded=false;
	//ComputeObjectiveValue();//initializes _labeling
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::_InitMove(ValueType rho,MoveDirection movedirection)
{
	_core_InitMoves(rho,movedirection);

	UpdateMarginals();
}

template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::FinalizeMove()
{
	_objectiveValue=ComputeObjectiveValue();
	_bInitializationNeeded=true;
};

//template<class InputIterator>
template<class GM,class ACC,class InputIterator>
void DynamicProgramming<GM,ACC,InputIterator>::IncreaseUnaryWeights(InputIteratorType begin,InputIteratorType end)
{
	exception_check((LabelType)abs(end-begin)==_storage.unaryFactors(_currentUnaryIndex).size(),"SumProdSequenceTRWSSolver::IncreaseUnaryWeights(): (end-begin)!=unaryFactor.size()");

	std::transform(begin,end,_storage.ufBegin(_currentUnaryIndex),_storage.ufBegin(_currentUnaryIndex),std::plus<ValueType>());
	std::transform(_currentUnaryFactor.begin(),_currentUnaryFactor.end(),begin,_currentUnaryFactor.begin(),plus2ndMul<ValueType>(1.0/_rho));
}

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::getPrevPWId()const
{
	if (_currentUnaryIndex >= _storage.size()) return NaN;

	  if (_moveDirection==Storage::Direct)
		  return (_currentUnaryIndex==0 ? NaN : _storage.pwForwardFactor(_currentUnaryIndex-1));
	  else
		  return (_currentUnaryIndex==_storage.size()-1 ? NaN : _storage.pwForwardFactor(_currentUnaryIndex));
}

template<class GM,class ACC,class InputIterator>
typename DynamicProgramming<GM,ACC,InputIterator>::IndexType
DynamicProgramming<GM,ACC,InputIterator>::getNextPWId()const
{
	if (_currentUnaryIndex >= (IndexType)_storage.size()) return NaN;

	  if (_moveDirection==Storage::Direct)
		  return (_currentUnaryIndex==_storage.size()-1 ? NaN : _storage.pwForwardFactor(_currentUnaryIndex));
	  else
		  return (_currentUnaryIndex==0 ? NaN : _storage.pwForwardFactor(_currentUnaryIndex-1));
}

template<class GM,class ACC,class InputIterator>
void MaxSumSolver<GM,ACC,InputIterator>::_SumUpBackwardEdges(UnaryFactor* pu, LabelType fixedLabel)const
{
	UnaryFactor& u=*pu;
	IndexType factorId=parent::getPrevPWId();
	OPENGM_ASSERT(factorId!=parent::NaN);

	if ((parent::_factorProperties.getFunctionType(factorId)==FunctionParameters<GM>::POTTS) && parent::_fastComputation)
	{
       if (fixedLabel<u.size())
		u[fixedLabel]-=parent::_factorProperties.getFunctionParameters(factorId)[0];//instead of adding everywhere the same we just subtract the difference
//       else
//    	transform_inplace(u.begin(),u.end(),std::bind2nd(std::plus<ValueType>(),parent::_factorProperties.getFunctionParameters(factorId)[0]));
	}else
	{
	const typename GM::FactorType& pwfactor=parent::_storage.masterModel()[factorId];

	OPENGM_ASSERT( (parent::_storage.varIndex(parent::_currentUnaryIndex)==pwfactor.variableIndex(0)) || (parent::_storage.varIndex(parent::_currentUnaryIndex)==pwfactor.variableIndex(1)));

	IndexType localVarIndx = (parent::_storage.varIndex(parent::_currentUnaryIndex)==pwfactor.variableIndex(0) ?  1 : 0);
	opengm::ViewFixVariablesFunction<GM> pencil(pwfactor,
			std::vector<opengm::PositionAndLabel<IndexType,LabelType> >(1,
					opengm::PositionAndLabel<IndexType,LabelType>(localVarIndx,
							fixedLabel)));

	for (LabelType j=0;j<u.size();++j)
		u[j]+=pencil(&j);
	}
}

//========================SumProdSolver  Implementation =============================================

template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_PushMessagesToVariable()
{
	LabelType srcsize=parent::_marginals[parent::_previous(parent::_currentUnaryIndex)].size();//check asserts of _previous first

	parent::_spst.resize(srcsize,parent::_currentUnaryFactor.size());

    //sum up for each pencil
	for (LabelType i=0;i<parent::_currentUnaryFactor.size();++i)
		parent::_currentUnaryFactor[i]+=_mul*log(std::accumulate(parent::_spst.beginTrg(&parent::_currentPWFactor[0],i),parent::_spst.endTrg(&parent::_currentPWFactor[0],i),ValueType(0.0)));//TODO: multiply by mul!
}

template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_UpdatePWAverage()
{
	std::transform(_unaryBuffer.begin(),_unaryBuffer.end(),parent::_marginals[parent::_currentUnaryIndex].begin(),
			       _unaryBuffer.begin(),std::plus<ValueType>());//adding logarithms of the right-hand side
	transform_inplace(_unaryBuffer.begin(),_unaryBuffer.end(),std::bind2nd(std::minus<ValueType>(),_getMarginalsLogNormalizer()));//normalize
	transform_inplace(_unaryBuffer.begin(),_unaryBuffer.end(),mulAndExp<ValueType>(_mul));

	LabelType srcsize=parent::_marginals[parent::_previous(parent::_currentUnaryIndex)].size();
	parent::_spst.resize(srcsize,parent::_currentUnaryFactor.size());

	//sum up for each pencil
	for (LabelType i=0;i<parent::_currentUnaryFactor.size();++i)
		_unaryBuffer[i]*=std::inner_product(parent::_spst.beginTrg(&parent::_currentPWFactor[0],i),
				           parent::_spst.endTrg(&parent::_currentPWFactor[0],i),
				           parent::_spst.beginTrg(&_copyPWfactor[0],i),
				           ValueType(0.0));

	_derivativeValue+=std::accumulate(_unaryBuffer.begin(),_unaryBuffer.end(),(ValueType)0.0);//sum up. It is already divided by rho and taken with the correct sign due to _copyPWfactor
}

template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_PushAndAverage()
{
	//move unaries to fwMessages
	parent::_PushMessagesToFactor();//updates _currentPWFactor[pencil i]+=_currentUnaryFactor[i]
	_InitCurrentUnaryBuffer(parent::_next(parent::_currentUnaryIndex));

	//clear bkMessages
	parent::_ClearMessages(&_unaryBuffer);//updates _currentPWFactor, that each pencil contains 0 and _currUnaryFactor = unaryFactor/rho + pencil normalization

	//exponentiate p/w factor
	_ExponentiatePWFactor();//updates _currentPWFactor - exponentiation in place

	_UpdatePWAverage();//here we suppose, that logPartition is already computed, as this has to be a backward move

	//sum up, logarithm and add
    _PushMessagesToVariable();// updates _currentUnaryFactor+=log(sum Exp(p/w pencil))

}

template<class GM,class ACC,class InputIterator>
typename SumProdSolver<GM,ACC,InputIterator>::ValueType
SumProdSolver<GM,ACC,InputIterator>::_GetAveragedUnaryFactors()
{
	ValueType unaryAverage=0.0;
	for (size_t i=0;i<parent::size();++i)
	{
		_unaryBuffer.resize(parent::_marginals[i].size());
		std::transform(parent::_marginals[i].begin(),parent::_marginals[i].end(),_unaryBuffer.begin(),std::bind2nd(std::minus<ValueType>(),_getMarginalsLogNormalizer()));
		transform_inplace(_unaryBuffer.begin(),_unaryBuffer.end(),mulAndExp<ValueType>(_mul));
		unaryAverage+=std::inner_product(_unaryBuffer.begin(),_unaryBuffer.end(),parent::_storage.unaryFactors(i).begin(),(ValueType)0.0);
	}
	return unaryAverage;
}

template<class GM,class ACC,class InputIterator>
typename SumProdSolver<GM,ACC,InputIterator>::ValueType
SumProdSolver<GM,ACC,InputIterator>::MoveBackGetDerivative()
{
if (parent::_bInitializationNeeded)
{
	parent::_InitReverseMoveBack();
}

_averagingFlag=true;
_derivativeValue=0.0;
 for (size_t i=0;i<parent::size()-1;++i)
 {
	 _PushAndAverage();
	 parent::_SumUpBufferToMarginals();
 }

 _derivativeValue+=_GetAveragedUnaryFactors();
 parent::FinalizeMove();
 _averagingFlag=false;
  _derivativeValue=(parent::GetObjectiveValue()-_derivativeValue)/parent::_rho;
 return _derivativeValue;
}

template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_Push()
{
	parent::_Push();

	//exponentiate p/w factor
	_ExponentiatePWFactor();//updates _currentPWFactor - exponentiation in place

	//sum up, logarithm and add
    _PushMessagesToVariable();// updates _currentUnaryFactor+=log(sum Exp(p/w pencil))
}

template <class T,class ACC> struct thresholdMulAndExp : std::unary_function <T,T> {
	thresholdMulAndExp(T threshold):_mul(ACC::template bop<T>(1.0,0.0) ? 1.0 : -1.0),_threshold(threshold){};
  T operator() (T x)
    {_buf=fabs(x); return (_buf >= _threshold ? 0.0 : exp(-_buf));}
private:
  T _mul;
  T _threshold;
  T _buf;
};


//exponentiates the temporary p/w factor in place
template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_ExponentiatePWFactor()
{
	transform_inplace(parent::_currentPWFactor.begin(),parent::_currentPWFactor.end(),thresholdMulAndExp<ValueType,ACC>(-log(std::numeric_limits<ValueType>::epsilon())));
}

template<class T,class InputIterator,class OutputIterator,class Comp >
T _MaxNormalize(InputIterator begin, InputIterator end, OutputIterator outBegin, T init,Comp comp)
{
	T max=*std::max_element(begin,end,comp);
	std::transform(begin,end,outBegin,std::bind2nd(std::minus<T>(),max));
	return init+max;
}

//uses _currentUnaryFactor as a buffer for computations
template<class GM,class ACC,class InputIterator>
typename SumProdSolver<GM,ACC,InputIterator>::ValueType
SumProdSolver<GM,ACC,InputIterator>::ComputeObjectiveValue()
{
	typename UnaryFactor::const_iterator begin=parent::_marginals[parent::_currentUnaryIndex].begin(),
						  end=parent::_marginals[parent::_currentUnaryIndex].end();
	 parent::_unaryTemp.resize(end-begin);
	 ValueType logPartition= parent::_rho*_MaxNormalize(begin,end,parent::_unaryTemp.begin(),(ValueType)0.0,ACC::template ibop<ValueType>);
	 std::transform(parent::_unaryTemp.begin(),parent::_unaryTemp.end(),parent::_unaryTemp.begin(),mulAndExp<ValueType>(_mul));
	 logPartition+=_mul*parent::_rho*(log(std::accumulate(parent::_unaryTemp.begin(),parent::_unaryTemp.end(),(ValueType)0.0)));
	return logPartition;
}

template<class GM,class ACC,class InputIterator>
void SumProdSolver<GM,ACC,InputIterator>::_makeLocalCopyOfPWFactor(LabelType trgsize)
{
	parent::_makeLocalCopyOfPWFactor(trgsize);
	if (_averagingFlag)
		_copyPWfactor=parent::_currentPWFactor;//!> optimization may be needed - instead of the memory reallocation just copy in it, as soon as enough space provided
}

template<class GM,class ACC,class InputIterator>
void  SumProdSolver<GM,ACC,InputIterator>::_InitCurrentUnaryBuffer(IndexType index)
{
 	parent::_InitCurrentUnaryBuffer(index);

	if (parent::_rho!=1.0) transform_inplace(parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.end(),std::bind2nd(std::multiplies<ValueType>(),1.0/parent::_rho));
}

};//namespace trws_base
} //namespace opengm

#endif /* ITERATIVESOLVERTRWS_H_ */
