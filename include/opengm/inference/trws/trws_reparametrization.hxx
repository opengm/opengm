#ifndef REPARAMETRIZATION_HXX
#define REPARAMETRIZATION_HXX
#include <valarray>
#include <iostream>
#include <map>
#include <opengm/inference/trws/trws_subproblemsolver.hxx>
#include <opengm/inference/inference.hxx>
#include <opengm/inference/trws/trws_base.hxx>
#include <opengm/inference/trws/utilities2.hxx>

#include <opengm/inference/auxiliary/lp_reparametrization.hxx>

namespace opengm {

namespace trws_base{

/*
 * Trivialization solver computes dual variables, which trivialize the input problem, making local decision globally consistent
 */

template<class GM,class ACC,class InputIterator>
class TrivializationSolver : protected MaxSumSolver<GM,ACC,InputIterator>
{
public:
	typedef MaxSumSolver<GM,ACC,InputIterator> parent;
	typedef typename parent::ValueType ValueType;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;

	typedef typename parent::InputIteratorType InputIteratorType;
	typedef typename parent::Storage Storage;
	typedef opengm::LPReparametrisationStorage<GM> DualStorage;
	typedef typename parent::MoveDirection MoveDirection;
	typedef typename parent::UnaryFactor UnaryFactor;
	typedef typename UnaryFactor::const_iterator const_uIterator;
	typedef typename parent::FactorProperties FactorProperties;

	typedef typename std::vector<bool> MaskType;
	typedef typename std::vector<MaskType>  ImmovableLabelingType;

	TrivializationSolver(Storage& primalstorage,
			             DualStorage& dualstorage,
			             const FactorProperties& fp,
			             bool fastComputations=true)
	:parent(primalstorage,fp,fastComputations),
	 _dualstorage(dualstorage){};
	void ForwardMove(MoveDirection direction=Storage::Direct){parent::InitMove(direction); parent::Move();};
	void BackwardMove(const MaskType* pmask=0);
	//void BackwardMove(const ImmovableLabelingType& immovableLabels);

	ValueType  GetObjectiveValue()const{return parent::GetObjectiveValue();}
private:
	void _PushBack();
	//void _PushBack(const MaskType& mask);
	IndexType _distanceFromStart();
	void _InitBackwardMoveBuffer(IndexType index);
	void _setDuals(IndexType index,typename SequenceStorage<GM>::MoveDirection moveDir,const_uIterator it);
	DualStorage& _dualstorage;
	MaskType _mask;
	IndexType _numberOfBoundaryTerms;

	// computation optimization

	//std::vector<ValueType> _multipliers;
};

//=======================TrivializationSolver implementation ===========================================
template<class GM,class ACC,class InputIterator>
typename TrivializationSolver<GM,ACC,InputIterator>::IndexType
TrivializationSolver<GM,ACC,InputIterator>::_distanceFromStart()
{
	if (parent::_moveDirection==Storage::Direct)
		return parent::_currentUnaryIndex;
	else
		return parent::size()-1-parent::_currentUnaryIndex;
}

template<class GM,class ACC,class InputIterator>
void TrivializationSolver<GM,ACC,InputIterator>::_InitBackwardMoveBuffer(IndexType index)
{
	assert(index < parent::_storage.size());
	parent::_currentUnaryIndex=index;
	parent::_currentUnaryFactor.resize(parent::_storage.unaryFactors(parent::_currentUnaryIndex).size());
	std::copy(parent::_marginals[parent::_currentUnaryIndex].
			begin(),parent::_marginals[parent::_currentUnaryIndex].end(),parent::_currentUnaryFactor.begin());

	_numberOfBoundaryTerms=std::count(_mask.begin(),_mask.end(),true);
}



template<class GM,class ACC,class InputIterator>
void TrivializationSolver<GM,ACC,InputIterator>::_PushBack()
{
	OPENGM_ASSERT(_mask.size()==parent::size());
	ValueType multiplier;
	if (_mask[parent::_currentUnaryIndex])
	{
     multiplier=((ValueType)_numberOfBoundaryTerms-1.0)/_numberOfBoundaryTerms;
     --_numberOfBoundaryTerms;
	}
    else
    {
     multiplier=1.0;
    }

	transform_inplace(parent::_currentUnaryFactor.begin(),
			parent::_currentUnaryFactor.end(),
			std::bind2nd(std::multiplies<ValueType>(),multiplier));

	std::transform(parent::_currentUnaryFactor.begin(),
			parent::_currentUnaryFactor.end(),
			parent::_marginals[parent::_currentUnaryIndex].begin(),
			parent::_currentUnaryFactor.begin(),
			std::minus<ValueType>());
	std::transform(parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.end(),
			parent::_storage.unaryFactors(parent::_currentUnaryIndex).begin(),
			parent::_currentUnaryFactor.begin(),std::plus<ValueType>());

	_setDuals(parent::_currentUnaryIndex,parent::_moveDirection,parent::_currentUnaryFactor.begin());

	parent::_PushMessagesToFactor();
	parent::_currentUnaryIndex=parent::_next(parent::_currentUnaryIndex);//instead of _InitCurrentUnaryBuffer(_next(_currentUnaryIndex));
	parent::_currentUnaryFactor.assign(parent::_storage.unaryFactors(parent::_currentUnaryIndex).size(),0.0);
	parent::_ClearMessages();

	transform_inplace(parent::_currentUnaryFactor.begin(),
			parent::_currentUnaryFactor.end(),
			std::bind2nd(std::multiplies<ValueType>(),-1.0));
	_setDuals(parent::_currentUnaryIndex,Storage::ReverseDirection(parent::_moveDirection),
			parent::_currentUnaryFactor.begin());

	std::transform(parent::_marginals[parent::_currentUnaryIndex].begin(),
			parent::_marginals[parent::_currentUnaryIndex].end(),
			parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.begin(),
			std::minus<ValueType>());
}

//template<class GM,class ACC,class InputIterator>
//void TrivializationSolver<GM,ACC,InputIterator>::_PushBack(const MaskType& mask)
//{
//	OPENGM_ASSERT(mask.size()==parent::_currentUnaryFactor.size());
//	_multipliers.resize(parent::_currentUnaryFactor.size());
//	bool decrease=false;
//
//	//std::cout << "_numberOfBoundaryTerms="<<_numberOfBoundaryTerms<<std::endl;
//
//	for (IndexType label=0;label<_multipliers.size();++label)
//	{
//		if (mask[label]) _multipliers[label]=1.0;
//		else
//		{
//			_multipliers[label]=((ValueType)_numberOfBoundaryTerms-1.0)/_numberOfBoundaryTerms;
//			decrease=true;
//		}
//	}
//
//	if (decrease)
//	{
//	 --_numberOfBoundaryTerms;
//	 decrease=false;
//	}
//
//	//std::cout << "_multipliers:" <<_multipliers<<std::endl;
//
//	transform(parent::_currentUnaryFactor.begin(),
//			parent::_currentUnaryFactor.end(),
//			_multipliers.begin(),parent::_currentUnaryFactor.begin(),
//			std::multiplies<ValueType>());
//
//	std::transform(parent::_currentUnaryFactor.begin(),
//			parent::_currentUnaryFactor.end(),
//			parent::_marginals[parent::_currentUnaryIndex].begin(),
//			parent::_currentUnaryFactor.begin(),
//			std::minus<ValueType>());
//	std::transform(parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.end(),
//			parent::_storage.unaryFactors(parent::_currentUnaryIndex).begin(),
//			parent::_currentUnaryFactor.begin(),std::plus<ValueType>());
//
//	_setDuals(parent::_currentUnaryIndex,parent::_moveDirection,parent::_currentUnaryFactor.begin());
//
//	parent::_PushMessagesToFactor();
//	parent::_currentUnaryIndex=parent::_next(parent::_currentUnaryIndex);//instead of _InitCurrentUnaryBuffer(_next(_currentUnaryIndex));
//	parent::_currentUnaryFactor.assign(parent::_storage.unaryFactors(parent::_currentUnaryIndex).size(),0.0);
//	parent::_ClearMessages();
//
//	transform_inplace(parent::_currentUnaryFactor.begin(),
//			parent::_currentUnaryFactor.end(),
//			std::bind2nd(std::multiplies<ValueType>(),-1.0));
//	_setDuals(parent::_currentUnaryIndex,Storage::ReverseDirection(parent::_moveDirection),
//			parent::_currentUnaryFactor.begin());
//
//	std::transform(parent::_marginals[parent::_currentUnaryIndex].begin(),
//			parent::_marginals[parent::_currentUnaryIndex].end(),
//			parent::_currentUnaryFactor.begin(),parent::_currentUnaryFactor.begin(),
//			std::minus<ValueType>());
//}


template<class GM,class ACC,class InputIterator>
void TrivializationSolver<GM,ACC,InputIterator>::BackwardMove(const MaskType* pmask)
{
	if (pmask==0)
		_mask.assign(parent::size(),true);
	else
		_mask=*pmask;

	parent::_moveDirection=Storage::ReverseDirection(parent::_moveDirection);
	if (parent::_moveDirection==Storage::Direct)
		_InitBackwardMoveBuffer(0);
	else
		_InitBackwardMoveBuffer(parent::size()-1);

	for (IndexType i=0;i<parent::size()-1;++i)//!> number of iterations is equal to a number of pairwise factors
		_PushBack(); //_Push(size()-i) - the current value of i is known, as _currentIndex

	parent::_bInitializationNeeded=true;
}

//template<class GM,class ACC,class InputIterator>
//void TrivializationSolver<GM,ACC,InputIterator>::BackwardMove(const ImmovableLabelingType& immovableLabels)
//{
//	_mask.assign(parent::size(),true);
//	parent::_moveDirection=Storage::ReverseDirection(parent::_moveDirection);
//
//	if (parent::_moveDirection==Storage::Direct)
//	{
//		//std::cout << "Direct"<<std::endl;
//		_InitBackwardMoveBuffer(0);
//	}
//	else
//	{
//		//std::cout << "Reverse"<<std::endl;
//		_InitBackwardMoveBuffer(parent::size()-1);
//	}
//
//	//for (IndexType i=0;i<parent::size()-1;++i)//!> number of iterations is equal to a number of pairwise factors
//	for (IndexType i=0;i<parent::size()-1;++i)//!> number of iterations is equal to a number of pairwise factors
//		_PushBack(immovableLabels[parent::_currentUnaryIndex]); //_Push(size()-i) - the current value of i is known, as _currentIndex
//
//	parent::_bInitializationNeeded=true;
//}

template<class GM,class ACC,class InputIterator>
void TrivializationSolver<GM,ACC,InputIterator>
::_setDuals(IndexType index,typename SequenceStorage<GM>::MoveDirection movedir,const_uIterator it)
 {
	IndexType pwId, varId;

	if (movedir==Storage::Direct)
	{
		pwId=parent::_storage.pwForwardFactor(index);
		varId=(parent::_storage.pwDirection(index)==Storage::Direct ? 0 : 1);
	}
	else
	{
		pwId=parent::_storage.pwForwardFactor(index-1);
		varId=(parent::_storage.pwDirection(index-1)==Storage::Direct ? 1 : 0);
	}
	std::pair<typename DualStorage::uIterator,typename DualStorage::uIterator> dualIt
	=_dualstorage.getIterators(pwId,varId);
	std::copy(it,it+(dualIt.second-dualIt.first),dualIt.first);
 };
}//namespace trws_base


template<class ValueType>
struct TRWS_Reparametrizer_Parameters
{
	bool fastComputations_;

	TRWS_Reparametrizer_Parameters(bool fastComputations=true):
		fastComputations_(fastComputations)
	{};
};

template<class Storage,class ACC>
class TRWS_Reparametrizer : public opengm::LPReparametrizer<typename Storage::GraphicalModelType, ACC>
{
public:
	typedef typename opengm::LPReparametrizer<typename Storage::GraphicalModelType, ACC> parent;
	typedef typename parent::GraphicalModelType GraphicalModelType;
	typedef typename GraphicalModelType::ValueType ValueType;
	typedef typename GraphicalModelType::IndexType IndexType;
	typedef typename GraphicalModelType::LabelType LabelType;

	typedef typename parent::RepaStorageType RepaStorageType;
	typedef typename parent::MaskType MaskType;
	typedef typename parent::ImmovableLabelingType ImmovableLabelingType;
	typedef typename parent::ReparametrizedGMType ReparametrizedGMType;

	typedef trws_base::TrivializationSolver<GraphicalModelType,ACC,typename std::vector<typename GraphicalModelType::ValueType>::const_iterator> SubSolverType;

	typedef TRWS_Reparametrizer_Parameters<ValueType> Parameter;
	typedef trws_base::FunctionParameters<GraphicalModelType> FunctionParametersType;

	TRWS_Reparametrizer(Storage& storage,const FunctionParametersType& fparams,const Parameter& params=Parameter());
	virtual ~TRWS_Reparametrizer();
	void reparametrize(const MaskType* pmask=0);
	void reparametrize(const ImmovableLabelingType& immovableLabeling);

private:
	Storage& _storage;
	std::vector<SubSolverType*> _subSolvers;
};

template<class Storage,class ACC>
TRWS_Reparametrizer<Storage,ACC>::~TRWS_Reparametrizer()
{
	std::for_each(_subSolvers.begin(),_subSolvers.end(),trws_base::DeallocatePointer<SubSolverType>);
}

template<class Storage,class ACC>
TRWS_Reparametrizer<Storage,ACC>::TRWS_Reparametrizer(Storage& storage,
		const FunctionParametersType& fparams,
		const Parameter& params):
		parent(storage.masterModel()),
		_storage(storage)
		{
	_subSolvers.resize(_storage.numberOfModels());

	for (size_t modelId=0;modelId<_subSolvers.size();++modelId)
	{
		_subSolvers[modelId]= new SubSolverType(_storage.subModel(modelId),parent::Reparametrization(),fparams,params.fastComputations_);
	}
}


template<class Storage,class ACC>
void TRWS_Reparametrizer<Storage,ACC>::reparametrize(const MaskType* pmask)
{

	MaskType mask(pmask!=0 ? *pmask : MaskType(_storage.masterModel().numberOfVariables(),true));
	OPENGM_ASSERT(mask.size()==_storage.masterModel().numberOfVariables());
	ValueType 	bound=0;
	MaskType sequenceMask;
	for (size_t i=0;i<_subSolvers.size();++i)
	{
		typename Storage::SubModel& model=_storage.subModel(i);
		sequenceMask.resize(model.size());
		for (IndexType localInd=0; localInd<sequenceMask.size();++localInd)
		{
			OPENGM_ASSERT(model.varIndex(localInd) < mask.size());
			sequenceMask[localInd]=mask[model.varIndex(localInd)];
		}

	_subSolvers[i]->ForwardMove();
	_subSolvers[i]->BackwardMove(&sequenceMask);
	bound+=_subSolvers[i]->GetObjectiveValue();
	}

}

//template<class Storage,class ACC>
//void TRWS_Reparametrizer<Storage,ACC>::reparametrize(const ImmovableLabelingType& immovableLabeling)
//{
//
//	//MaskType mask(pmask!=0 ? *pmask : MaskType(_storage.masterModel().numberOfVariables(),true));
//	OPENGM_ASSERT(immovableLabeling.size()==_storage.masterModel().numberOfVariables());
//	ValueType 	bound=0;
//	ImmovableLabelingType sequenceLabeling;
//	for (size_t i=0;i<_subSolvers.size();++i)
//	{
//		typename Storage::SubModel& model=_storage.subModel(i);
//		sequenceLabeling.resize(model.size());
//		for (IndexType localInd=0; localInd<sequenceLabeling.size();++localInd)
//		{
//			OPENGM_ASSERT(model.varIndex(localInd) < immovableLabeling.size());
//			sequenceLabeling[localInd]=immovableLabeling[model.varIndex(localInd)];
//		}
//
//	//std::cout << "ForwardMove: ";
//	_subSolvers[i]->ForwardMove();
//	//std::cout << "BackwardMove: ";
//	_subSolvers[i]->BackwardMove(sequenceLabeling);
//	bound+=_subSolvers[i]->GetObjectiveValue();
//	}
//
//}

template<class Storage,class ACC>
void TRWS_Reparametrizer<Storage,ACC>::reparametrize(const ImmovableLabelingType& immovableLabeling)
{
	OPENGM_ASSERT(immovableLabeling.size()==_storage.masterModel().numberOfVariables());
	reparametrize();

	typedef typename parent::RepaStorageType::uIterator uIterator;

	const typename Storage::GraphicalModelType& gm=_storage.masterModel();
	for (IndexType factorID=0;factorID < gm.numberOfFactors();++factorID)
	{
		if (gm[factorID].numberOfVariables()<2) continue;
		/*
		 * Make zero potentials for immovable labels
		 */

		for (IndexType localVarID=0;localVarID<gm[factorID].numberOfVariables();++localVarID)
		{
			std::pair<uIterator,uIterator> it=parent::Reparametrization().getIterators(factorID,localVarID);
			IndexType globalVarID=gm[factorID].variableIndex(localVarID);
			typename MaskType::const_iterator labIt=immovableLabeling[globalVarID].begin();
			for (;it.first!=it.second;++it.first)
				if (*labIt++) *it.first=0;
		}

		/*
		 * Make reparametrized pairwise factors non-negative
		 */

		if (gm[factorID].numberOfVariables()!=2) throw std::runtime_error("TRWS_Reparametrizer<Storage,ACC>::reparametrize(): factors of order higher than 2 are not supported!");

		std::vector<IndexType> labeling(2);
		for (IndexType localVarID=0;localVarID<gm[factorID].numberOfVariables();++localVarID)
		{
			std::pair<uIterator,uIterator> it=parent::Reparametrization().getIterators(factorID,localVarID);
			uIterator it_begin=it.first;
			IndexType globalVarID=gm[factorID].variableIndex(localVarID);
			typename MaskType::const_iterator labIt=immovableLabeling[globalVarID].begin();
			ValueType res=ACC::template neutral<ValueType>();

			for (;it.first!=it.second;++it.first)
			  if (!(*labIt++))
			  {
				  IndexType otherVarID=(localVarID==0 ? 1 : 0);
				  labeling[localVarID]=it.first-it_begin;
				  for (LabelType label=0;label<gm.numberOfLabels(otherVarID);++label)
				  {
					  labeling[otherVarID]=label;

					  ValueType res1=parent::Reparametrization().getFactorValue(factorID,labeling.begin());
					  ACC::op(res,res1,res);
				  }
				  *it.first-=res;
			  }
		}
	}

}



//============ LP reparametrization to TRWS reparametrization ===========================
template<class GM>
void LPtoDecompositionStorage(const LPReparametrisationStorage<GM>& lpRepa, trws_base::DecompositionStorage<GM>* ptrwsRepa)
{
	OPENGM_ASSERT(&lpRepa.graphicalModel() == &ptrwsRepa->masterModel());

	typedef typename LPReparametrisationStorage<GM>::uIterator uIterator;
	typedef typename GM::ValueType ValueType;
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef typename trws_base::DecompositionStorage<GM> DecompositionStorage;

	std::vector<ValueType> repaUnary;
	//for all variables (and related unary factors)
	for (IndexType varId=0;varId<lpRepa.graphicalModel().numberOfVariables();++varId)// all variables
	{ const typename DecompositionStorage::SubVariableListType& varList=ptrwsRepa->getSubVariableList(varId);

	  if (varList.size()==1) continue;

	  // compute common part - the sum of all potentials
	  repaUnary.resize(lpRepa.graphicalModel().numberOfLabels(varId));
	  for (LabelType label=0;label<repaUnary.size();++label)
	  {
		  repaUnary[label]=lpRepa.getVariableValue(varId,label);
	  }

	  transform_inplace(repaUnary.begin(),repaUnary.end(),std::bind2nd(std::multiplies<ValueType>(),1.0/varList.size()));//!> repaUnary:=repaUnary/numberTrees

	  //for all submodels
	  for(typename DecompositionStorage::SubVariableListType::const_iterator modelIt=varList.begin();
			  	  	  	  	  	  	  	  	  	 modelIt!=varList.end();++modelIt) //all related models
	  {
		  typename DecompositionStorage::SubModel& subModel=ptrwsRepa->subModel(modelIt->subModelId_);
		  typename DecompositionStorage::SubModel::UnaryFactor::iterator uit_begin=subModel.ufBegin(modelIt->subVariableId_);
		  typename DecompositionStorage::SubModel::UnaryFactor::iterator uit_end  =subModel.ufEnd(modelIt->subVariableId_);
		  //unary=repaUnary/numberTrees
		  std::copy(repaUnary.begin(),repaUnary.end(),uit_begin);
		  //add only potentials belonging to the submodel
//		  std::pair<uIterator,uIterator> repaIt;
		  const typename LPReparametrisationStorage<GM>::UnaryFactor* prepaUF;
		  if (modelIt->subVariableId_ < subModel.size()-1)
			  {
			    IndexType pwId=subModel.pwForwardFactor(modelIt->subVariableId_);
			    if (lpRepa.graphicalModel()[pwId].variableIndex(0)==varId)
			    	prepaUF=&lpRepa.get(pwId,0);
			    else prepaUF=&lpRepa.get(pwId,1);

	            std::transform(uit_begin,uit_end,prepaUF->begin(),uit_begin,std::plus<ValueType>());
			  }
		if (modelIt->subVariableId_ >0)
			  {
				  IndexType pwId=subModel.pwForwardFactor(modelIt->subVariableId_-1);
				  if (lpRepa.graphicalModel()[pwId].variableIndex(0)==varId)
					  prepaUF=&lpRepa.get(pwId,0);
				  else prepaUF=&lpRepa.get(pwId,1);

				  std::transform(uit_begin,uit_end,prepaUF->begin(),uit_begin,std::plus<ValueType>());
			  }

	  }
	}
}


} //namespace opengm
#endif
