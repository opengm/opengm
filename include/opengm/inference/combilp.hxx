/*
 * combiLP.hxx
 *
 *  Created on: Sep 16, 2013
 *      Author: bsavchyn
 */

#ifndef COMBILP_HXX_
#define COMBILP_HXX_
#include <opengm/graphicalmodel/graphicalmodel_manipulator.hxx>
#include <opengm/inference/lpcplex.hxx>
#include <opengm/inference/auxiliary/lp_reparametrization.hxx>
#include <opengm/inference/trws/output_debug_utils.hxx>

namespace opengm{

namespace combilp_base{

template<class FACTOR>
void MakeFactorVariablesTrue(const FACTOR& f,std::vector<bool>* pmask)
{
	for (typename FACTOR::VariablesIteratorType it=f.variableIndicesBegin();
			it!=f.variableIndicesEnd();++it)
		(*pmask)[*it]=true;
}

template<class GM>
void DilateMask(const GM& gm,typename GM::IndexType varId,std::vector<bool>* pmask)
{
	typename GM::IndexType numberOfFactors=gm.numberOfFactors(varId);
	for (typename GM::IndexType localFactorId=0;localFactorId<numberOfFactors;++localFactorId)
	{
		const typename GM::FactorType& f=gm[gm.factorOfVariable(varId,localFactorId)];
		if (f.numberOfVariables()>1)
			MakeFactorVariablesTrue(f,pmask);
	}
}

/*
 * inmask and poutmask should be different objects!
 */
template<class GM>
void DilateMask(const GM& gm,const std::vector<bool>& inmask, std::vector<bool>* poutmask)
{
	*poutmask=inmask;
	for (typename GM::IndexType varId=0;varId<inmask.size();++varId)
		if (inmask[varId]) DilateMask(gm,varId,poutmask);

}

template<class GM>
bool LabelingMatching(const std::vector<typename GM::LabelType>& labeling1,const std::vector<typename GM::LabelType>& labeling2,
		const std::vector<bool>& mask,std::list<typename GM::IndexType>* presult)
{
	OPENGM_ASSERT(labeling1.size()==mask.size());
	OPENGM_ASSERT(labeling2.size()==mask.size());
	presult->clear();
	for (typename GM::IndexType varId=0;varId<mask.size();++varId)
		if ((mask[varId]) && (labeling1[varId]!=labeling2[varId]))
				presult->push_back(varId);

	return presult->empty();
}

template<class GM>
void GetMaskBoundary(const GM& gm,const std::vector<bool>& mask,std::vector<bool>* boundmask)
{
	boundmask->assign(mask.size(),false);
	for (typename GM::IndexType varId=0;varId<mask.size();++varId)
	{
		if (!mask[varId]) continue;

		typename GM::IndexType numberOfFactors=gm.numberOfFactors(varId);
		for (typename GM::IndexType localFactorId=0;localFactorId<numberOfFactors;++localFactorId)
		{
			if ((*boundmask)[varId]) break;

			const typename GM::FactorType& f=gm[gm.factorOfVariable(varId,localFactorId)];
			if (f.numberOfVariables()>1)
			{
				for (typename GM::FactorType::VariablesIteratorType it=f.variableIndicesBegin();
						it!=f.variableIndicesEnd();++it)
					if (!mask[*it])
					{
						(*boundmask)[varId]=true;
						break;
					}
			}
		}
	}
}

//template<class LPREPARAMETRIZERPARAMETERS>
struct CombiLP_base_Parameter{

	CombiLP_base_Parameter(size_t maxNumberOfILPCycles=100,
			bool verbose=false,
			const std::string& reparametrizedModelFileName="",//will not be saved if empty
			bool singleReparametrization=true,
			bool saveProblemMasks=false,
			std::string maskFileNamePre=""):
				maxNumberOfILPCycles_(maxNumberOfILPCycles),
				verbose_(verbose),
				reparametrizedModelFileName_(reparametrizedModelFileName),
				singleReparametrization_(singleReparametrization),
				saveProblemMasks_(saveProblemMasks),
				maskFileNamePre_(maskFileNamePre)
	{};
	virtual ~CombiLP_base_Parameter(){};
	size_t maxNumberOfILPCycles_;
	bool verbose_;
	std::string reparametrizedModelFileName_;
	bool singleReparametrization_;
	bool saveProblemMasks_;
	std::string maskFileNamePre_;

#ifdef TRWS_DEBUG_OUTPUT
	  virtual void print(std::ostream& fout)const
	  {
			fout <<"maxNumberOfILPCycles="<<maxNumberOfILPCycles_<<std::endl;
			fout <<"verbose"<<verbose_<<std::endl;
			fout <<"reparametrizedModelFileName="<<reparametrizedModelFileName_<<std::endl;
			fout <<"singleReparametrization="<<singleReparametrization_<<std::endl;
			fout <<"saveProblemMasks="<<saveProblemMasks_<<std::endl;
			fout <<"maskFileNamePre="<<maskFileNamePre_<<std::endl;
	  }
#endif
};



template<class GM, class ACC, class LPREPARAMETRIZER>//TODO: remove default ILP solver
class CombiLP_base
{
public:
	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;

	OPENGM_GM_TYPE_TYPEDEFS;

	typedef CombiLP_base_Parameter Parameter;
	typedef LPREPARAMETRIZER ReparametrizerType;
	typedef typename ReparametrizerType::MaskType MaskType;

	typedef typename opengm::GraphicalModelManipulator<typename ReparametrizerType::ReparametrizedGMType> GMManipulatorType;

	typedef LPCplex<typename GMManipulatorType::MGM, Minimizer> LPCPLEX;//TODO: move to template parameters

	CombiLP_base(LPREPARAMETRIZER& reparametrizer, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			, std::ostream& fout=std::cout
#endif
	);
	virtual ~CombiLP_base(){};

	const GraphicalModelType& graphicalModel() const { return _lpparametrizer->graphicalModel(); }

	InferenceTermination infer(MaskType& mask,const std::vector<LabelType>& lp_labeling);

	InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	{
		out = _labeling;
		return opengm::NORMAL;
	}

	ValueType value() const{return _value;};
	ValueType bound() const{return _bound;}

	void ReparametrizeAndSave();
private:
	void _Reparametrize(typename ReparametrizerType::ReparametrizedGMType* pgm,const MaskType& mask);
	InferenceTermination _PerformILPInference(GMManipulatorType& modelManipulator,std::vector<LabelType>* plabeling);
	Parameter _parameter;
	ReparametrizerType& _lpparametrizer;
	std::vector<LabelType> _labeling;
	ValueType _value;
	ValueType _bound;
#ifdef TRWS_DEBUG_OUTPUT
	std::ostream& _fout;
#endif
};

template<class GM, class ACC, class LPREPARAMETRIZER>
CombiLP_base<GM,ACC,LPREPARAMETRIZER>::CombiLP_base(LPREPARAMETRIZER& reparametrizer, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
		, std::ostream& fout
#endif
)
:  _parameter(param)
  ,_lpparametrizer(reparametrizer)
  ,_labeling(_lpparametrizer.graphicalModel().numberOfVariables(),std::numeric_limits<LabelType>::max())
  ,_value(ACC::template neutral<ValueType>())
  ,_bound(ACC::template ineutral<ValueType>())
#ifdef TRWS_DEBUG_OUTPUT
  ,_fout(param.verbose_ ? fout : *OUT::nullstream::Instance())//(fout)
#endif
  {
  };

template<class GM, class ACC, class LPREPARAMETRIZER>
InferenceTermination CombiLP_base<GM,ACC,LPREPARAMETRIZER>::_PerformILPInference(GMManipulatorType& modelManipulator,std::vector<LabelType>* plabeling)
{
	InferenceTermination terminationILP=NORMAL;
	modelManipulator.buildModifiedSubModels();

	std::vector<std::vector<LabelType> > submodelLabelings(modelManipulator.numberOfSubmodels());
    for (size_t modelIndex=0;modelIndex<modelManipulator.numberOfSubmodels();++modelIndex)
    {
    	const typename GMManipulatorType::MGM& model=modelManipulator.getModifiedSubModel(modelIndex);
    	submodelLabelings[modelIndex].resize(model.numberOfVariables());
    	typename LPCPLEX::Parameter param;
    	param.integerConstraint_=true;
    	LPCPLEX ilpSolver(model,param);
    	terminationILP=ilpSolver.infer();

    	if ((terminationILP!=NORMAL) && (terminationILP!=CONVERGENCE))
      		return terminationILP;
      	else
    	 ilpSolver.arg(submodelLabelings[modelIndex]);
    }

    modelManipulator.modifiedSubStates2OriginalState(submodelLabelings,*plabeling);
    return terminationILP;
}

template<class GM, class ACC, class LPREPARAMETRIZER>
InferenceTermination CombiLP_base<GM,ACC,LPREPARAMETRIZER>::infer(MaskType& mask,const std::vector<LabelType>& lp_labeling)
{
#ifdef TRWS_DEBUG_OUTPUT
    if (!_parameter.singleReparametrization_)
    	_fout << "Applying reparametrization for each ILP run ..."<<std::endl;
    else
    	_fout << "Applying a single uniform reparametrization..."<<std::endl;

    _fout <<"Switching to ILP."<<std::endl;
#endif

	bool startILP=true;
	typename ReparametrizerType::ReparametrizedGMType gm;
    bool reparametrizedFlag=false;
    InferenceTermination terminationId=TIMEOUT;

    for (size_t i=0;(startILP && (i<_parameter.maxNumberOfILPCycles_));++i)
	{

#ifdef TRWS_DEBUG_OUTPUT
		_fout << "Subproblem "<<i<<" size="<<std::count(mask.begin(),mask.end(),true)<<std::endl;
#endif

		MaskType boundmask(mask.size());
		GetMaskBoundary(_lpparametrizer.graphicalModel(),mask,&boundmask);

#ifdef TRWS_DEBUG_OUTPUT
		if (_parameter.saveProblemMasks_)
		{
		 OUT::saveContainer(std::string(_parameter.maskFileNamePre_+"-mask-"+trws_base::any2string(i)+".txt"),mask.begin(),mask.end());
		 OUT::saveContainer(std::string(_parameter.maskFileNamePre_+"-boundmask-"+trws_base::any2string(i)+".txt"),boundmask.begin(),boundmask.end());
		}
#endif

		if (_parameter.singleReparametrization_ && (!reparametrizedFlag) )
		{
#ifdef TRWS_DEBUG_OUTPUT
			_fout << "Reparametrizing..."<<std::endl;
#endif
			_Reparametrize(&gm,MaskType(mask.size(),true));
			reparametrizedFlag=true;
		}
		else if (!_parameter.singleReparametrization_)
		{
#ifdef TRWS_DEBUG_OUTPUT
			_fout << "Reparametrizing..."<<std::endl;
#endif
			_Reparametrize(&gm,mask);
		}

		OPENGM_ASSERT(mask.size()==gm.numberOfVariables());

		GMManipulatorType modelManipulator(gm,GMManipulatorType::DROP);
		modelManipulator.unlock();
		modelManipulator.freeAllVariables();
		for (IndexType varId=0;varId<mask.size();++varId)
			if (mask[varId]==0) modelManipulator.fixVariable(varId,lp_labeling[varId]);
		modelManipulator.lock();

		InferenceTermination terminationILP;
		std::vector<LabelType> labeling;
		terminationILP=_PerformILPInference(modelManipulator,&labeling);
		if ((terminationILP!=NORMAL) && (terminationILP!=CONVERGENCE))
		{
			_labeling=lp_labeling;
#ifdef TRWS_DEBUG_OUTPUT
			_fout << "ILP solver failed to solve the problem. LP solver results will be saved." <<std::endl;
#endif
			return terminationILP;
		}

#ifdef TRWS_DEBUG_OUTPUT
		_fout <<"Boundary size="<<std::count(boundmask.begin(),boundmask.end(),true)<<std::endl;
#endif

		std::list<IndexType> result;
		if (LabelingMatching<GM>(lp_labeling,labeling,boundmask,&result))
		{
			startILP=false;
			_labeling=labeling;
			_value=_bound=_lpparametrizer.graphicalModel().evaluate(_labeling);
			terminationId=NORMAL;
#ifdef TRWS_DEBUG_OUTPUT
			_fout <<"Solved! Optimal energy="<<value()<<std::endl;
#endif
		}
		else
		{
#ifdef TRWS_DEBUG_OUTPUT
			_fout <<"Adding "<<result.size()<<" nodes."<<std::endl;
			if (_parameter.saveProblemMasks_)
			  OUT::saveContainer(std::string(_parameter.maskFileNamePre_+"-added-"+trws_base::any2string(i)+".txt"),result.begin(),result.end());
#endif
			for (typename std::list<IndexType>::const_iterator it=result.begin();it!=result.end();++it)
				DilateMask(gm,*it,&mask);
		}
	}

	return terminationId;
}


template<class GM, class ACC, class LPREPARAMETRIZER>
void CombiLP_base<GM,ACC,LPREPARAMETRIZER>::
_Reparametrize(typename ReparametrizerType::ReparametrizedGMType* pgm,const MaskType& mask)
{
	_lpparametrizer.reparametrize(&mask);
	_lpparametrizer.getReparametrizedModel(*pgm);
}

template<class GM, class ACC, class LPREPARAMETRIZER>
void CombiLP_base<GM,ACC,LPREPARAMETRIZER>::
ReparametrizeAndSave()
{
    typename ReparametrizerType::ReparametrizedGMType gm;
    _Reparametrize(&gm,MaskType(_lpparametrizer.graphicalModel().numberOfVariables(),true));
	store_into_explicit(gm, _parameter.reparametrizedModelFileName_);
}

}//namespace combilp_base  =========================================================================

template<class LPSOLVERPARAMETERS,class REPARAMETRIZERPARAMETERS>
struct CombiLP_Parameter : public combilp_base::CombiLP_base_Parameter
{
	typedef combilp_base::CombiLP_base_Parameter parent;
	CombiLP_Parameter(LPSOLVERPARAMETERS lpsolverParameter=LPSOLVERPARAMETERS(),
			REPARAMETRIZERPARAMETERS repaParameter=REPARAMETRIZERPARAMETERS(),
			size_t maxNumberOfILPCycles=100,
			bool verbose=false,
			bool saveReparametrizedModel=false,
			const std::string& reparametrizedModelFileName="",
			bool singleReparametrization=true,
			bool saveProblemMasks=false,
			std::string maskFileNamePre=""):
				parent(maxNumberOfILPCycles,
						verbose,
						reparametrizedModelFileName,
						singleReparametrization,
						saveProblemMasks,
						maskFileNamePre),
				lpsolverParameter_(lpsolverParameter)
	{};
	LPSOLVERPARAMETERS lpsolverParameter_;
	REPARAMETRIZERPARAMETERS repaParameter_;

#ifdef TRWS_DEBUG_OUTPUT
	  void print(std::ostream& fout)const
	  {
		  parent::print(fout);
		  fout << "== lpsolverParameters: =="<<std::endl;
		  lpsolverParameter_.print(fout);
	  }
#endif
};


template<class GM, class ACC, class LPSOLVER>//TODO: remove default ILP solver
class CombiLP : public Inference<GM, ACC>
{
public:
	typedef typename LPSOLVER::ReparametrizerType ReparametrizerType;
	typedef combilp_base::CombiLP_base<GM,ACC,ReparametrizerType> BaseType;

	typedef ACC AccumulationType;
	typedef GM GraphicalModelType;

	OPENGM_GM_TYPE_TYPEDEFS;
	typedef VerboseVisitor<CombiLP<GM, ACC,LPSOLVER> > VerboseVisitorType;
	typedef TimingVisitor<CombiLP<GM, ACC,LPSOLVER> > TimingVisitorType;
	typedef EmptyVisitor< CombiLP<GM, ACC,LPSOLVER> > EmptyVisitorType;

	typedef CombiLP_Parameter<typename LPSOLVER::Parameter,typename ReparametrizerType::Parameter> Parameter;
	typedef typename ReparametrizerType::MaskType MaskType;
	typedef typename BaseType::GMManipulatorType GMManipulatorType;

	typedef LPCplex<typename GMManipulatorType::MGM, ACC> LPCPLEX;//TODO: move to template parameters

	CombiLP(const GraphicalModelType& gm, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
			, std::ostream& fout=std::cout
#endif
	);
	virtual ~CombiLP(){if (_plpparametrizer!=0) delete _plpparametrizer;};
	std::string name() const{ return "CombiLP"; }
	const GraphicalModelType& graphicalModel() const { return _lpsolver.graphicalModel(); }
	InferenceTermination infer()
	{
		EmptyVisitorType vis;
		return infer(vis);
	};

	template<class VISITOR> InferenceTermination infer(VISITOR & visitor);

	InferenceTermination arg(std::vector<LabelType>& out, const size_t = 1) const
	{
		out = _labeling;
		return opengm::NORMAL;
	}
	virtual ValueType bound() const{return _bound;};
	virtual ValueType value() const{return _value;};
private:
	Parameter _parameter;
	LPSOLVER _lpsolver;
	ReparametrizerType* _plpparametrizer;
	BaseType _base;
	std::vector<LabelType> _labeling;
	ValueType _value;
	ValueType _bound;
#ifdef TRWS_DEBUG_OUTPUT
	std::ostream& _fout;
#endif
};

template<class GM, class ACC, class LPSOLVER>
CombiLP<GM,ACC,LPSOLVER>::CombiLP(const GraphicalModelType& gm, const Parameter& param
#ifdef TRWS_DEBUG_OUTPUT
		, std::ostream& fout
#endif
)
: _parameter(param)
  ,_lpsolver(gm,param.lpsolverParameter_
#ifdef TRWS_DEBUG_OUTPUT
		  ,(param.lpsolverParameter_.verbose_ ? fout : *OUT::nullstream::Instance())//fout
#endif
  )
  ,_plpparametrizer(_lpsolver.getReparametrizer(_parameter.repaParameter_))//TODO: parameters of the reparametrizer come here
  ,_base(*_plpparametrizer, param
  #ifdef TRWS_DEBUG_OUTPUT
  		,fout
  #endif
  )
  ,_labeling(gm.numberOfVariables(),std::numeric_limits<LabelType>::max())
  ,_value(_lpsolver.value())
  ,_bound(_lpsolver.bound())
#ifdef TRWS_DEBUG_OUTPUT
  ,_fout(param.verbose_ ? fout : *OUT::nullstream::Instance())//(fout)
#endif
  {
#ifdef TRWS_DEBUG_OUTPUT
	  _fout << "Parameters of the "<< name() <<" algorithm:"<<std::endl;
	  param.print(_fout);
#endif
  };

template<class GM, class ACC, class LPSOLVER>
template<class VISITOR>
InferenceTermination CombiLP<GM,ACC,LPSOLVER>::infer(VISITOR & visitor)
{
#ifdef TRWS_DEBUG_OUTPUT
	_fout <<"Running LP solver "<<_lpsolver.name()<<std::endl;
#endif
	visitor.begin(*this, value(), bound());

	_lpsolver.infer();
	_value=_lpsolver.value();
	_bound=_lpsolver.bound();
	_lpsolver.arg(_labeling);

	visitor(*this, value(), bound());

	std::vector<LabelType> labeling_lp;
	MaskType initialmask;
	_plpparametrizer->reparametrize();
	//_plpparametrizer->getArcConsistency(&initialmask,&labeling_lp);
	_lpsolver.getTreeAgreement(initialmask,&labeling_lp);

#ifdef TRWS_DEBUG_OUTPUT
	_fout <<"Energy of the labeling consistent with the arc consistency ="<<_lpsolver.graphicalModel().evaluate(labeling_lp)<<std::endl;
	_fout <<"Arc inconsistent set size ="<<std::count(initialmask.begin(),initialmask.end(),false)<<std::endl;
#endif

#ifdef TRWS_DEBUG_OUTPUT
	_fout << "Trivializing."<<std::endl;
#endif

#ifdef	WITH_HDF5
	if (_parameter.reparametrizedModelFileName_.compare("")!=0)
	{
#ifdef	TRWS_DEBUG_OUTPUT
	_fout << "Saving reparametrized model..."<<std::endl;
#endif
	visitor(*this, value(), bound());
	_base.ReparametrizeAndSave();
	visitor(*this, value(), bound());
	}
#endif

	if (std::count(initialmask.begin(),initialmask.end(),false)==0)
		return NORMAL;

	trws_base::transform_inplace(initialmask.begin(),initialmask.end(),std::logical_not<bool>());

	MaskType mask;
	combilp_base::DilateMask(_lpsolver.graphicalModel(),initialmask,&mask);

	InferenceTermination terminationVal=_base.infer(mask,labeling_lp);
	if ( (terminationVal==NORMAL) || (terminationVal==CONVERGENCE) )
	{
		 _value=_base.value();
		 _bound=_base.bound();
		 _base.arg(_labeling);
	}

	visitor.end(*this, value(), bound());
	return terminationVal;
}


}


#endif /* COMBILP_HXX_ */
