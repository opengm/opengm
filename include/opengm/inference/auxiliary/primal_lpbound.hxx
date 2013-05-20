/*
 * primal_lpbound.hxx
 *
 *  Created on: Jan 31, 2013
 *      Author: bsavchyn
 */

#ifndef PRIMAL_LPBOUND_HXX_
#define PRIMAL_LPBOUND_HXX_
#include <limits>
#include <algorithm>
#include <opengm/inference/auxiliary/transportationsolver.hxx>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/inference/trws/utilities2.hxx>

namespace opengm
{

using trws_base::FactorWrapper;
using trws_base::VariableToFactorMapping;

/*
 * Restriction of the code:   unary and pairwise factors only. Extension to higher order possible (in future)
 *
 * Usage:
 * PrimalLPBound<GM, Minimizer> bound(gm);
 * std::vector<double> var(10, 0.1);
 * bound.setVariable(varIndex,var.begin());
 * bound.setVariable(.... )
 * ...
 * double optimalValue=bound.getTotalValue();
 */

template<class ValueType>
struct PrimalLPBound_Parameter
{
	PrimalLPBound_Parameter(ValueType relativePrecision,
			                size_t maxIterationNumber)
	:relativePrecision_(relativePrecision),
	 maxIterationNumber_(maxIterationNumber){};

	ValueType relativePrecision_;
	size_t maxIterationNumber_;
};

//! [class primallpbound]
/// PrimalLPBound - estimating primal local polytope bound and feasible primal solution
/// for the local polytope relaxation of the MRF energy minimization problem
/// Based on the paper:
/// B. Savchynskyy, S. Schmidt
/// Getting Feasible Variable Estimates From Infeasible Ones: MRF Local Polytope Study. arXiv:1210.4081 Submitted Oct. 2012
///
/// it provides:
/// * primal relaxed solution and bound for the local polytope relaxation of the MRF energy minimization problem
///
///
/// Corresponding author: Bogdan Savchynskyy
///
///\ingroup inference

template <class GM,class ACC>
class PrimalLPBound
{
public:
	typedef TransportSolver::TransportationSolver<ACC,FactorWrapper<typename GM::FactorType> > Solver;
    typedef typename Solver::floatType ValueType;
    typedef std::vector<ValueType> UnaryFactor;
    typedef typename GM::IndexType IndexType;
    typedef typename GM::LabelType LabelType;

    static const IndexType InvalidIndex;
    static const ValueType ValueTypeNan;

    typedef PrimalLPBound_Parameter<ValueType> Parameter;

	PrimalLPBound(const GM& gm,const Parameter& param=Parameter(Solver::floatTypeEps,Solver::defaultMaxIterationNumber));

	template<class ValueIterator>
	void setVariable(IndexType var, ValueIterator inputBegin);
	template<class ValueIterator>
	void getVariable(IndexType var, ValueIterator outputBegin);//memory has to be allocated in advance

	ValueType getTotalValue(); // calls getFactorValue() and getVariableValue() and add them. Buffered
	ValueType getFactorValue(IndexType factorId);//pairwise factor factorId. Buffering of the current value is performed
	ValueType getVariableValue(IndexType varId); //inner product. Buffered
	template<class Matrix>
	ValueType getFactorVariable(IndexType factorId, Matrix& matrix); //pairwise factor factorId, buffering of the solution is performed

	void ResetBuffer(){_bufferedValues(_gm.numberOfFactors(),ValueTypeNan); _totalValue=ValueTypeNan;}//reset buffer, if you changed potentials of gm and want to take this fact into account
	bool IsValueBuffered(IndexType factorId)const{OPENGM_ASSERT(factorId<_bufferedValues.size()); return (_bufferedValues[factorId] != ValueTypeNan);}
	bool IsFactorVariableBuffered(IndexType factorId)const{return _lastActiveSolver==factorId;}
	static void CheckDuplicateUnaryFactors(const GM& gm);
private:
	void _checkPWFactorID(IndexType factorId,const std::string& message_prefix=std::string());
	const GM& _gm;
	Solver _solver;
	std::vector<UnaryFactor> _unaryFactors;
	VariableToFactorMapping<GM> _mapping;

	std::vector<ValueType> _bufferedValues;
	IndexType _lastActiveSolver;
	ValueType _totalValue;
};

template <class GM,class ACC>
void PrimalLPBound<GM,ACC>::CheckDuplicateUnaryFactors(const GM& gm)
{
	std::vector<IndexType> numOfunaryFactors(gm.numberOfVariables(),0);
	for (IndexType factorId=0;factorId<gm.numberOfFactors();++factorId)
	{
		if (gm[factorId].numberOfVariables()!=1)
			continue;

		numOfunaryFactors[gm[factorId].variableIndex(0)]++;
	}

	IndexType moreCount=std::count_if(numOfunaryFactors.begin(),numOfunaryFactors.end(),std::bind2nd(std::greater<IndexType>(),1));
	if (moreCount!=0)
			throw std::runtime_error("PrimalLPBound::CheckDuplicateUnaryFactors: all variables must have not more then a single associated unary factor!");
}

template <class GM,class ACC>
const typename PrimalLPBound<GM,ACC>::IndexType PrimalLPBound<GM,ACC>::InvalidIndex=std::numeric_limits<IndexType>::max();

template <class GM,class ACC>
const typename PrimalLPBound<GM,ACC>::ValueType PrimalLPBound<GM,ACC>::ValueTypeNan=std::numeric_limits<ValueType>::max();

template <class GM,class ACC>
PrimalLPBound<GM,ACC>::PrimalLPBound(const GM& gm,const Parameter& param):
_gm(gm),
_solver(
#ifdef TRWS_DEBUG_OUTPUT
		std::cerr,
#endif
		param.relativePrecision_,param.maxIterationNumber_),
_unaryFactors(gm.numberOfVariables()),
_mapping(gm),
_bufferedValues(gm.numberOfFactors(),ValueTypeNan),
_lastActiveSolver(InvalidIndex),
_totalValue(ValueTypeNan)
{
	CheckDuplicateUnaryFactors(gm);
	//allocating memory for the unary factors
	for (size_t i=0;i<_unaryFactors.size();++i)
		_unaryFactors[i].assign(_gm.numberOfLabels(i),0);
}

template <class GM,class ACC>
template<class Iterator>
void PrimalLPBound<GM,ACC>::setVariable(IndexType var, Iterator inputBegin)
{
	OPENGM_ASSERT(var < _gm.numberOfVariables());
	_totalValue=ValueTypeNan;
	std::copy(inputBegin,inputBegin+_unaryFactors[var].size(),_unaryFactors[var].begin());

	//making invalid factors connected to the variable var
	IndexType numOfFactors=_gm.numberOfFactors(var);
	for (IndexType i=0;i<numOfFactors;++i)
	{
		IndexType factorId=_gm.factorOfVariable(var,i);
		OPENGM_ASSERT(factorId < _gm.numberOfFactors() );
		_bufferedValues[factorId] = ValueTypeNan;
	}
}

template <class GM,class ACC>
template<class Iterator>
void PrimalLPBound<GM,ACC>::getVariable(IndexType var, Iterator outputBegin)
{
	OPENGM_ASSERT(var < _gm.numberOfVariables());
	std::copy(_unaryFactors[var].begin(),_unaryFactors[var].end(),outputBegin);
}

template <class GM,class ACC>
void PrimalLPBound<GM,ACC>::_checkPWFactorID(IndexType factorId, const std::string& message_prefix)
{
	OPENGM_ASSERT(factorId < _gm.numberOfFactors());
	if (_gm[factorId].numberOfVariables() !=2 )
		std::runtime_error(message_prefix + "Function can be applied to second order factors only!");
}

template <class GM,class ACC>
typename PrimalLPBound<GM,ACC>::ValueType PrimalLPBound<GM,ACC>::getFactorValue(IndexType factorId)
{
	_checkPWFactorID(factorId,"PrimalLPBound::getFactorValue(): ");

	if (_bufferedValues[factorId] == ValueTypeNan)
	{
	const typename GM::FactorType& factor=_gm[factorId];
	IndexType var0=factor.variableIndex(0),
			  var1=factor.variableIndex(1);
	_solver.Init(_unaryFactors[var0].size(),_unaryFactors[var1].size(),FactorWrapper<typename GM::FactorType>(factor));
	_bufferedValues[factorId]=_solver.Solve(_unaryFactors[var0].begin(),_unaryFactors[var1].begin());
	_lastActiveSolver=factorId;
	}

	return _bufferedValues[factorId];
}

template <class GM,class ACC>
template<class Matrix>
typename PrimalLPBound<GM,ACC>::ValueType PrimalLPBound<GM,ACC>::getFactorVariable(IndexType factorId,  Matrix& matrix)
{
	_checkPWFactorID(factorId,"PrimalLPBound::getFactorVariable(): ");

	if (_lastActiveSolver!=factorId)
		getFactorValue(factorId);

	return _solver.GetSolution(&matrix);
}

template <class GM,class ACC>
typename PrimalLPBound<GM,ACC>::ValueType PrimalLPBound<GM,ACC>::getVariableValue(IndexType varId)
{
	OPENGM_ASSERT(varId < _gm.numberOfVariables());
	OPENGM_ASSERT(varId < _unaryFactors.size());
	IndexType factorId=_mapping(varId);
	OPENGM_ASSERT(_mapping(varId) < _gm.numberOfFactors());
	if (factorId==VariableToFactorMapping<GM>::InvalidIndex)
		return (ValueType)0;

	if (_bufferedValues[factorId] != ValueTypeNan)
		return _bufferedValues[factorId];

	ValueType sum=0;
	const UnaryFactor& uf=_unaryFactors[varId];
	OPENGM_ASSERT(_gm.numberOfLabels(varId)==uf.size());
	OPENGM_ASSERT(_gm.numberOfLabels(varId)>0);
	const typename GM::FactorType& f=_gm[factorId];
	for (LabelType i=0;i<uf.size();++i)
    	sum+=uf[i]*f(&i);

	_bufferedValues[factorId]=sum;
    return sum;
}

template <class GM,class ACC>
typename PrimalLPBound<GM,ACC>::ValueType PrimalLPBound<GM,ACC>::getTotalValue()
{
	if (_totalValue==ValueTypeNan)
	{
	_totalValue=0;
	for (IndexType factorId=0;factorId<_gm.numberOfFactors();++factorId)
	{
		const typename GM::FactorType& f=_gm[factorId];
		switch (f.numberOfVariables())
		{
		case 1:	_totalValue+=getVariableValue(f.variableIndex(0)); break;
		case 2: _totalValue+=getFactorValue(factorId);break;
		default: throw std::runtime_error("PrimalLPBound::getTotalValue(): Only factors of order <= 2 are supported!");
		}
	}
	}
	return _totalValue;
}

}
#endif /* PRIMAL_LPBOUND_HXX_ */
