/*
 * smallobjects.h
 *
 *  Created on: Jul 28, 2010
 *      Author: bsavchyn
 */

#ifndef UTILITIES2_HXX_
#define UTILITIES2_HXX_
#include <functional>
#include <numeric>
#include <string>
#include <stdexcept>
#include <cmath>
#include <sstream>

#ifdef TRWS_DEBUG_OUTPUT
#include "output_debug_utils.hxx"
#endif

namespace opengm {
namespace trws_base{

template<class AnyType>
std::string any2string(const AnyType& any)
{
	  std::stringstream out;
	  out << any;
	  return out.str();
}

template<class GM>
class VariableToFactorMapping
{
public:
	typedef typename GM::IndexType IndexType;
	static const IndexType InvalidIndex;
	VariableToFactorMapping(const GM& gm);
	IndexType operator() (IndexType var)const{OPENGM_ASSERT(var < _mapping.size()); return _mapping[var];}
	bool EachVariableHasUnaryFactor()const{return (_numberOfUnaryFactors==_mapping.size());}
	IndexType numberOfUnaryFactors()const{return _numberOfUnaryFactors;}
private:
	std::vector<IndexType> _mapping;
	IndexType _numberOfUnaryFactors;
};

template <class GM>
const typename VariableToFactorMapping<GM>::IndexType VariableToFactorMapping<GM>::InvalidIndex=std::numeric_limits<IndexType>::max();

template<class GM>
VariableToFactorMapping<GM>::VariableToFactorMapping(const GM& gm):
_mapping(gm.numberOfVariables(),InvalidIndex)//invalid index
{
	for (IndexType i=0;i<gm.numberOfFactors();++i)
		if (gm[i].numberOfVariables() == 1)
		{ if (_mapping[gm[i].variableIndex(0)]==InvalidIndex)
			_mapping[gm[i].variableIndex(0)]=i;
		  else std::runtime_error("VariableToFactorMapping::VariableToFactorMapping() : duplicate unary factor!");
		}

	_numberOfUnaryFactors=std::count_if(_mapping.begin(),_mapping.end(),std::bind2nd(std::not_equal_to<IndexType>(),InvalidIndex));
}

template<class FACTOR>
class FactorWrapper
{
public:
	typedef typename FACTOR::ValueType ValueType;
	typedef typename FACTOR::LabelType LabelType;
	FactorWrapper(const FACTOR& f):_f(f){};
	ValueType operator () (LabelType l1, LabelType l2)const
	 {LabelType lab[]={l1,l2}; return _f(lab);}
private:
	const FACTOR& _f;
};


template < class InputIterator, class UnaryOperator >
InputIterator transform_inplace ( InputIterator first, InputIterator last, UnaryOperator op )
{
  while (first != last)
  {
    *first= op(*first);
    ++first;
  }

  return first;
}

inline void exception_check (bool condition,const std::string& str)
{
 if (!condition) throw std::runtime_error(str);
}

template < class Matrix,class OutputIterator, class Pseudo2DArray>
OutputIterator copy_transpose( const Matrix* src,size_t totalsize,OutputIterator outBegin,size_t rowlength, Pseudo2DArray& arr2d)
{
	exception_check(totalsize%rowlength == 0,"copy_transpose(): totalsize%rowlength != 0");

	arr2d.resize(rowlength,totalsize/rowlength);
	for (size_t i=0;i<rowlength;++i)
		outBegin=std::copy(arr2d.beginSrc(src,i),arr2d.endSrc(src,i),outBegin);

	return outBegin;
}


template <class T> struct plus2ndMul : std::binary_function <T,T,T> {
	plus2ndMul(T mul):_mul(mul){};
  T operator() (T x, T y) const
    {return x+y*_mul;}
private:
  T _mul;
};

template <class T> struct mul2ndPlus : std::binary_function <T,T,T> {
	mul2ndPlus(T add):_add(add){};
  T operator() (T x, T y) const
    {return x*(y+_add);}
private:
  T _add;
};

template <class T> struct mulAndExp : std::unary_function <T,T> {
	mulAndExp(T mul):_mul(mul){};
  T operator() (T x) const
    {return ::exp(_mul*x);}
private:
  T _mul;
};


template <class T> struct make0ifless : std::unary_function <T,T> {
	make0ifless(T threshold):_threshold(threshold){};
  T operator() (T x) const
    {return (x < _threshold ? 0.0 : x);}
private:
  T _threshold;
};

template <class T> struct minusminus : std::binary_function <T,T,T> {
  T operator() (const T& x, const T& y) const
    {return y-x;}
};

template <class T> struct plusplusConst : std::binary_function <T,T,T>
{
	plusplusConst(const T& constant):_constant(constant) {};
	T operator() (const T& x, const T& y) const
    {return x+y+_constant;}
private:
	T _constant;
};


template <class T> struct maximum : std::binary_function <T,T,T> {
  T operator() (const T& x, const T& y) const
    {return std::max(x,y);}
};


 template<class T>
 class srcIterator
 {
	public:
	 typedef T value_type;
	 typedef std::forward_iterator_tag iterator_category;
	 typedef void difference_type;
	 typedef T* pointer;
	 typedef T& reference;
	 srcIterator():_pbin(0),_pindex(0){};
	 srcIterator(const srcIterator<T>& it):_pbin(it._pbin),_pindex(it._pindex){};
	 srcIterator(T* pbin, size_t* pindex):_pbin(pbin),_pindex(pindex){};
	 //T& operator * (){return (*_pbin)[*_pindex];}
	 T& operator * (){return *(_pbin+(*_pindex));}
	 srcIterator& operator ++(){++_pindex; return *this;}
	 srcIterator operator ++(int){srcIterator it(*this);++_pindex; return it;}
	 bool operator != (const srcIterator& it)const{return ((it._pbin!=_pbin)||(it._pindex!=_pindex));};
	 bool operator == (const srcIterator& it)const{return !(*this!=it);}
	 srcIterator& operator +=(size_t offset){_pindex+=offset; return (*this);}
	private:
	 T* _pbin;
	 size_t* _pindex;
 };

 template<class T>
 srcIterator<T> operator + (const srcIterator<T>& it,size_t offset){srcIterator<T> result=it; return (result+=offset);}

 template<class T>
 class Pseudo2DArray
 {
 public:
	 typedef srcIterator<const T> const_srciterator;
	 typedef const T* const_trgiterator;
	 typedef srcIterator<T> srciterator;
	 typedef T* trgiterator;

	 Pseudo2DArray(size_t srcsize=0,size_t trgsize=0):_srcsize(srcsize),_trgsize(trgsize){_setupindex();};

 	inline void resize(size_t srcsize,size_t trgsize);

 	const_srciterator beginSrc(const T* pbin,size_t src) {assert(src<_srcsize); return const_srciterator(pbin,&_index[src*_trgsize]);};
  	const_srciterator endSrc(const T* pbin,size_t src){return beginSrc(pbin,src)+_trgsize;};

 	const_trgiterator beginTrg(const T* pbin,size_t trg){assert(trg<_trgsize); return pbin+trg*_srcsize;};
	const_trgiterator endTrg(const T* pbin,size_t trg){return beginTrg(pbin,trg)+_srcsize;}

 	srciterator beginSrcNC(T* pbin,size_t src) {assert(src<_srcsize); return srciterator(pbin,&_index[src*_trgsize]);};
  	srciterator endSrcNC(T* pbin,size_t src){return beginSrcNC(pbin,src)+_trgsize;};

 	trgiterator beginTrgNC(T* pbin,size_t trg){assert(trg<_trgsize); return pbin+trg*_srcsize;};
	trgiterator endTrgNC(T* pbin,size_t trg){return beginTrgNC(pbin,trg)+_srcsize;}

 private:
	void _setupindex();
 	size_t _srcsize;
 	size_t _trgsize;
 	std::vector<size_t> _index;
 };

 template<class T>
 void Pseudo2DArray<T>::resize(size_t srcsize,size_t trgsize)
 {
  if ((srcsize==_srcsize) && (trgsize==_trgsize))
 	 return;

 	_srcsize=srcsize;
 	_trgsize=trgsize;
 	_setupindex();
 };

 template<class T>
 void Pseudo2DArray<T>::_setupindex()
 {
  _index.assign(_srcsize*_trgsize,0);
  if (_index.empty())
		return;
  size_t* _pindex=&_index[0];
  for (size_t src=0;src<_srcsize;++src)
   	for (size_t trg=0;trg<_trgsize;++trg)
 		 *(_pindex++)=trg*_srcsize+src;
 };


 template<class Object>
 void DeallocatePointer(Object* p)
 {
	if (p!=0) delete p;
 }


};
}

#endif /* SMALLOBJECTS_H_ */
