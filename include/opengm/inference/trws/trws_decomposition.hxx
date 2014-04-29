#ifndef TRWS_DECOMPOSITION_HXX_
#define TRWS_DECOMPOSITION_HXX_
#include <iostream>
#include <list>
#include <vector>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <opengm/graphicalmodel/decomposition/graphicalmodeldecomposition.hxx>
#include <opengm/inference/trws/utilities2.hxx>

#ifdef TRWS_DEBUG_OUTPUT
#include <opengm/inference/trws/output_debug_utils.hxx>
#endif

namespace opengm {
namespace trws_base{

#ifdef TRWS_DEBUG_OUTPUT
using OUT::operator <<;
#endif

template<class GM>
class Decomposition
{
public:
	typedef typename GM::IndexType IndexType;
	typedef typename GM::LabelType LabelType;
	typedef std::vector<IndexType> IndexList;
	typedef opengm::GraphicalModelDecomposition::SubVariable SubVariable;
	typedef opengm::GraphicalModelDecomposition::SubVariableListType SubVariableListType;
	Decomposition(const GM& gm,IndexType numSubModels=0)//!< numSubModels - ESTIMATED number of submodels to optimize memory allocation
	:_numberOfModels(0),_gm(gm)
	{	// Reserve memory
		_variableLists.reserve(numSubModels);
		_pwFactorLists.reserve(numSubModels);
	};
	virtual ~Decomposition()=0;

	virtual IndexType 		 getNumberOfSubModels()const{return _numberOfModels;}
	virtual const IndexList& getVariableList(IndexType subModelId)const {return _variableLists[subModelId];}
	virtual const IndexList& getFactorList(IndexType subModelId)const {return _pwFactorLists[subModelId];}

#ifdef TRWS_DEBUG_OUTPUT
	virtual void PrintTestData(std::ostream& fout);
#endif

	virtual void ComputeVariableDecomposition(std::vector<SubVariableListType>* plist)const;

	static void CheckUnaryFactors(const GM& gm);//!< checks whether all variables have corresp. unary factor with the same index and vice versa
	static void CheckDuplicateUnaryFactors(const GM& gm);
	/*static*/ void CheckForIsolatedNodes(const GM& gm);
protected:
	typedef std::pair<IndexType,IndexType> Edge;//first=factorId, second=neighborNodeId
	typedef std::list<Edge> EdgeList;
	typedef std::vector<EdgeList> NodeList;

	IndexType _numberOfModels;
	std::vector<IndexList> _variableLists;
	std::vector<IndexList> _pwFactorLists;
	const GM& _gm;

	IndexType _addSubModel();
	void _addSubFactor(const IndexType& factorId);
	void _addSubVariable(const IndexType& variableId);
	static void _CreateNodeList(const GM& gm,NodeList* pnodeList);
};

/*
 * So far oriented to 2-nd order factors only!
 */
template<class GM>
class MonotoneChainsDecomposition : public Decomposition<GM>
{
public:
	typedef Decomposition<GM> parent;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;
	typedef typename parent::IndexList IndexList;
	typedef typename parent::SubVariable SubVariable;
	typedef typename parent::SubVariableListType SubVariableListType;

	MonotoneChainsDecomposition(const GM& gm,IndexType numSubModels=0);//!< numSubModels - ESTIMATED number of submodels to optimize memory allocation
protected:
	void _GetMaximalMonotoneSequence(typename parent::NodeList* pnodesList,IndexType start);
};

template<class GM>
class GridDecomposition : public Decomposition<GM>
{
public:
	typedef Decomposition<GM> parent;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;
	typedef typename parent::IndexList IndexList;
	typedef typename parent::SubVariable SubVariable;
	typedef typename parent::SubVariableListType SubVariableListType;

	GridDecomposition(const GM& gm,IndexType numSubModels=0);//!< numSubModels - ESTIMATED number of submodels to optimize memory allocation
	IndexType xsize()const{return _xsize;}
	IndexType ysize()const{return _ysize;}
private:
	IndexType _xsize, _ysize;
protected:
	void _computeGridSizes();
	void _CheckGridModel();
	void _initDecompositionLists();

	IndexType _xysize()const{return _xsize*_ysize;}
	IndexType _pwrowsize()const{return 2*_xsize-1;}
	IndexType _pwIndexRow(IndexType x,IndexType y)const;//!> returns an index of a row pairwise factor places to the right to var (x,y)
	IndexType _pwIndexCol(IndexType x,IndexType y)const;//!> returns an index of a column pairwise factor places to the down to var (x,y)
	IndexType _varIndex(IndexType x,IndexType y)const{return x+_xsize*y;}
	void _getRow(IndexType y,IndexList* plist)const;//!> returns indexes of variables in the row <y>
	void _getCol(IndexType x,IndexList* plist)const;//!> returns indexes of variables in the column <y>
	void _getPWRow(IndexType y, IndexList* plist)const;//!> return indexes of pairwise factors in the row <y>
	void _getPWCol(IndexType x,IndexList* plist)const;//!> return indexes of pairwise factors in the column <x>
};

template<class GM>
class EdgeDecomposition : public Decomposition<GM>
{
public:
	typedef Decomposition<GM> parent;
	typedef typename parent::IndexType IndexType;
	typedef typename parent::LabelType LabelType;
	typedef typename parent::IndexList IndexList;
	typedef typename parent::SubVariable SubVariable;
	typedef typename parent::SubVariableListType SubVariableListType;

	EdgeDecomposition(const GM& gm):parent(gm)
	 {
		parent::CheckUnaryFactors(gm);
		parent::CheckDuplicateUnaryFactors(gm);
		parent::_numberOfModels=gm.numberOfFactors()-gm.numberOfVariables();//!> this should be a number of pairwise factors
		//bild variable and factor lists
		parent::_variableLists.resize(parent::_numberOfModels,IndexList(2,(IndexType)0));
		parent::_pwFactorLists.resize(parent::_numberOfModels,IndexList(1,(IndexType)0));

		IndexType pwFid=0;
		for (IndexType fId=0;fId<gm.numberOfFactors();++fId)
		{
			if (gm[fId].numberOfVariables()==1) continue;
			if ((gm[fId].numberOfVariables()>2) || (gm[fId].numberOfVariables()==0))
				std::runtime_error("EdgeDecomposition<GM>::EdgeDecomposition(): Only factors of order 1 or 2 are supported!");

			//factor of order 2:
			parent::_variableLists[pwFid][0]=gm[fId].variableIndex(0);
			parent::_variableLists[pwFid][1]=gm[fId].variableIndex(1);
			parent::_pwFactorLists[pwFid][0]=fId;
			++pwFid;
		}
	}
};


#ifdef TRWS_DEBUG_OUTPUT
template <class SubFactor>
struct printSubFactor
{
	printSubFactor(std::ostream& out):_out(out){};
	void operator()(const SubFactor& a)
	  {
		  _out << "("<<a.subModelId_ <<","<< a.subFactorId_ <<")"<<", ";
	  }

private:
	std::ostream& _out;
};
#endif

#ifdef TRWS_DEBUG_OUTPUT
template <class SubVariable>
struct printSubVariable
{
	printSubVariable(std::ostream& out):_out(out){};
	void operator()(const SubVariable& a)
	  {
		  _out << "("<<a.subModelId_ <<","<< a.subVariableId_ <<")"<<", ";
	  }

private:
	std::ostream& _out;
};
#endif
//-------------------------IMPLEMENTATION------------------------------------------------

template<class GM>
Decomposition<GM>::~Decomposition<GM>()
{}

#ifdef TRWS_DEBUG_OUTPUT
template<class GM>
void Decomposition<GM>::PrintTestData(std::ostream& fout)
{
 fout <<"_numberOfModels;" << _numberOfModels<<std::endl;
 fout <<"_variableLists:"<<_variableLists<<std::endl;
 fout <<"_pwFactorLists:"<<_pwFactorLists<<std::endl;
}
#endif

template<class GM>
MonotoneChainsDecomposition<GM>::MonotoneChainsDecomposition(const GM& gm,IndexType numSubModels)
:parent(gm,numSubModels)
{	parent::CheckDuplicateUnaryFactors(gm);
	parent::CheckForIsolatedNodes(gm);

	typename parent::NodeList nodeList(gm.numberOfVariables());
	parent::_CreateNodeList(gm,&nodeList);

	for (IndexType start=0;start<nodeList.size();++start)
	 while (!nodeList[start].empty())
	 {   parent::_addSubModel();
		 _GetMaximalMonotoneSequence(&nodeList,(IndexType)start);
	 }
}

template<class GM>
GridDecomposition<GM>::GridDecomposition(const GM& gm,IndexType numSubModels)
:parent(gm,numSubModels)
 {
	//estimate xsize and ysize
	_computeGridSizes();
	parent::_numberOfModels=_xsize+_ysize;
	//bild variable and factor lists
	_initDecompositionLists();
 }

template<class Factor>
bool dependsOnVariable(const Factor& f,typename Factor::IndexType varId)
{
	return (std::find(f.variableIndicesBegin(),f.variableIndicesEnd(),varId) != f.variableIndicesEnd());
}

template<class GM>
void GridDecomposition<GM>::_computeGridSizes()
{
	IndexType numberOfVars=parent::_gm.numberOfVariables();
	IndexType numTotal=parent::_gm.numberOfFactors();
	std::vector<IndexType> ind;
	for (IndexType f=numberOfVars;f<numTotal;++f)
	{
		std::vector<IndexType> ind(parent::_gm[f].numberOfVariables());
		if (ind.size()!=2)
			throw std::runtime_error("GridDecomposition<GM>::_computeGridSizes():Incorrect grid structure! : only pairwise factors are supported !=0");
		parent::_gm[f].variableIndices(ind.begin());
		if (ind[1]<=ind[0])
			throw std::runtime_error("GridDecomposition<GM>::_computeGridSizes():Incorrect grid structure! : pairwise factors should be oriented from smaller to larger variable indices !=0");

		if (ind[1]-ind[0]!=1)
		{
			_xsize=ind[1]-ind[0];
			_ysize=numberOfVars/_xsize;
			if (numberOfVars%_xsize !=0)
				throw std::runtime_error("GridDecomposition<GM>::_computeGridSizes():Incorrect grid structure! : numberOfVars%xsize !=0");
			break;
		}else if (f==numTotal-1)
		{
			_xsize=numberOfVars;
			_ysize=1;
			break;
		};

	};
	_CheckGridModel();
};

template<class GM>
void GridDecomposition<GM>::_CheckGridModel()
{
	bool incorrect=false;
	//check vertical structure
	for (IndexType y=0;y<_ysize;++y)
	 for (IndexType x=0;x<_xsize;++x)
	 {
		if (y<_ysize-1)
		{
		 IndexType ind=_pwIndexCol(x,y);
		 if (!dependsOnVariable(parent::_gm[ind],_varIndex(x,y)) || !dependsOnVariable(parent::_gm[ind],_varIndex(x,y+1)) )
			incorrect=true;
		};

		if ((x<_xsize-1))
		{
		IndexType ind=_pwIndexRow(x,y);
		if (!dependsOnVariable(parent::_gm[ind],_varIndex(x,y)) || !dependsOnVariable(parent::_gm[ind],_varIndex(x+1,y)))
			incorrect=true;
		}

		if (incorrect)
		throw std::runtime_error("GridDecomposition::_CheckGridModel():Incorrect grid structure!");
	 };
};


template<class GM>
void GridDecomposition<GM>::_initDecompositionLists()
{
	parent::_variableLists.resize(parent::_numberOfModels);
	parent::_pwFactorLists.resize(parent::_numberOfModels);
	for (IndexType x=0;x<_xsize;++x)
	{
		_getCol(x,&parent::_variableLists[x]);
		_getPWCol(x,&parent::_pwFactorLists[x]);
	}

	for (IndexType y=0;y<_ysize;++y)
	{
		_getRow(y,&parent::_variableLists[_xsize+y]);
		_getPWRow(y,&parent::_pwFactorLists[_xsize+y]);
	};
}

//make the vector of nodes with lists of edges. Each edge is present only once - in the list of the node with the smaller index
template<class GM>
void Decomposition<GM>::_CreateNodeList(const GM & gm,NodeList* pnodeList)
{
	NodeList& varList=*pnodeList;
	varList.resize(gm.numberOfVariables());
	for (IndexType factorId=0;factorId<gm.numberOfFactors();++factorId)
	{
		if (gm[factorId].numberOfVariables()>2)
			throw std::runtime_error("CreateEdgeList(): Only factors up to order 2 are supported!");

		if (gm[factorId].numberOfVariables()==1) continue;
		std::vector<IndexType> varIndices(gm[factorId].variableIndicesBegin(),gm[factorId].variableIndicesEnd());
		if (varIndices[0] < varIndices[1])
		 varList[varIndices[0]].push_back(std::make_pair(factorId,varIndices[1]));
		else
		 varList[varIndices[1]].push_back(std::make_pair(factorId,varIndices[0]));
	}
}

template<class GM>
typename Decomposition<GM>::IndexType Decomposition<GM>::_addSubModel()
{
	_variableLists.push_back(IndexList());
	_pwFactorLists.push_back(IndexList());
	_numberOfModels++;
	return IndexType(_numberOfModels-1);
};

template<class GM>
void Decomposition<GM>::_addSubFactor(const IndexType& factorId)
	{
	 _pwFactorLists[_numberOfModels-1].push_back(factorId);
	}

template<class GM>
void Decomposition<GM>::_addSubVariable(const IndexType& variableId)
{
	_variableLists[_numberOfModels-1].push_back(variableId);
}

template<class GM>
void MonotoneChainsDecomposition<GM>::_GetMaximalMonotoneSequence(typename parent::NodeList* pnodeList,IndexType start)
{
 assert(start < pnodeList->size());
 typename parent::NodeList& nodeList=*pnodeList;
 if (!nodeList[start].empty())
	 parent::_addSubVariable(start);
 else return;

 while ( !nodeList[start].empty() )
 {
	typename parent::EdgeList::iterator it= nodeList[start].begin();
	parent::_addSubVariable(it->second);
	parent::_addSubFactor(it->first);
	IndexType tmp=it->second;
	nodeList[start].erase(it);
	start=tmp;
 }

}

template<class GM>
void Decomposition<GM>::CheckUnaryFactors(const GM& gm)
{
 bool error=false;
	for (IndexType factorId=0;factorId<gm.numberOfFactors();++factorId)
	{
		std::vector<IndexType> varIndices(gm[factorId].variableIndicesBegin(),gm[factorId].variableIndicesEnd());
		if (gm[factorId].numberOfVariables()==1)
		{
		 if ( (factorId < gm.numberOfVariables()) &&  (varIndices[0]==factorId))
			 continue;
		 else error=true;
		}else if (factorId < gm.numberOfVariables())
			error=true;

		if (error)
		 throw std::runtime_error("Decomposition<GM>::CheckUnaryFactors(): Each variable has to have a unique unary factor, which moreover has the same index!");
	}
}

template<class GM>
void Decomposition<GM>::CheckDuplicateUnaryFactors(const GM& gm)
{
	std::vector<IndexType> numOfunaryFactors(gm.numberOfVariables(),(IndexType)0);
	for (IndexType factorId=0;factorId<gm.numberOfFactors();++factorId)
	{
		if (gm[factorId].numberOfVariables()!=1)
			continue;

		numOfunaryFactors[gm[factorId].variableIndex(0)]++;
	}

	IndexType oneCount=std::count(numOfunaryFactors.begin(),numOfunaryFactors.end(),(IndexType)1);
	exception_check(oneCount==numOfunaryFactors.size(),"Decomposition::CheckDuplicateUnaryFactors: all variables must have a unique associated unary factor!");
}

template<class GM>
void Decomposition<GM>::CheckForIsolatedNodes(const GM& gm)
{
	for (IndexType varId=0;varId<gm.numberOfVariables();++varId)
	{
	  bool isolatedNode=true;
	  for (IndexType localId=0;localId<gm.numberOfFactors(varId);++localId)
	  {
		  if (gm[gm.factorOfVariable(varId,localId)].numberOfVariables()>1)
		         isolatedNode=false;
	  }
	  if (isolatedNode==true)
	  {
		  _addSubModel();
		  _addSubVariable(varId);
//TODO:TEST		  throw std::runtime_error("Decomposition<GM>::CheckForIsolatedNodes(): Procesing of isolated nodes is not supported!");
	  }
	}
}

template<class GM>
void Decomposition<GM>::ComputeVariableDecomposition(std::vector<SubVariableListType>* plist)const
{
	plist->resize(_gm.numberOfVariables());
	for (IndexType modelId=0;modelId<_numberOfModels;++modelId)
		for (IndexType varId=0;varId<_variableLists[modelId].size();++varId)
			(*plist)[_variableLists[modelId][varId]].push_back(SubVariable(modelId,varId));
}

template<class GM>
typename GridDecomposition<GM>::IndexType
GridDecomposition<GM>::_pwIndexRow(IndexType x,IndexType y)const//!> returns an index of a row pairwise factor places to the right to var (x,y)
{
	assert(x<_xsize-1);
	assert(y<_ysize);
	if ((y==_ysize-1) && (x!=0)) return _pwIndexRow(0,y) + x;
	return _xysize()+y*_pwrowsize()+2*x;
};
template<class GM>
typename GridDecomposition<GM>::IndexType
GridDecomposition<GM>::_pwIndexCol(IndexType x,IndexType y)const//!> returns an index of a column pairwise factor places to the down to var (x,y)
{
	if (x==_xsize-1) return _pwIndexCol(x-1,y)+1;
	return _pwIndexRow(x,y)+1;
};

template<class GM>
void GridDecomposition<GM>::
_getRow(IndexType y,IndexList* plist)const//!> returns indexes of variables in the row <y>
{
	plist->resize(_xsize);
	(*plist)[0]=_varIndex(0,y);
	for (IndexType i=1;i<_xsize;++i)
		(*plist)[i]=(*plist)[i-1]+1;
};

template<class GM>
void GridDecomposition<GM>::
_getCol(IndexType x,IndexList* plist)const//!> returns indexes of variables in the column <y>
{
	plist->resize(_ysize);
	(*plist)[0]=_varIndex(x,0);
	for (IndexType i=1;i<_ysize;++i)
		(*plist)[i]=(*plist)[i-1]+_xsize;
};

template<class GM>
void GridDecomposition<GM>::
_getPWRow(IndexType y, IndexList* plist)const//!> return indexes of pairwise factors in the row <y>
{
	plist->resize(_xsize-1);
	if (_xsize<=1)
		return;
	(*plist)[0]=_pwIndexRow(0,y);
	IndexType step=2;
	if (y==_ysize-1) step=1;
	for (IndexType i=1;i<_xsize-1;++i)
		 (*plist)[i]=(*plist)[i-1]+step;
};

template<class GM>
void GridDecomposition<GM>::
_getPWCol(IndexType x,IndexList* plist)const//!> return indexes of pairwise factors in the column <x>
{
	plist->resize(_ysize-1);
	if (_ysize<=1)
		return;

	(*plist)[0]=_pwIndexCol(x,0);
	for (IndexType i=1;i<_ysize-1;++i)
		 (*plist)[i]=(*plist)[i-1]+_pwrowsize();
};


}//namespace trws_base
}//namespace opengm

#endif /* DECOMPOSITIONTRWS_H_ */
