#ifndef PRIMALSOLVER_H_
#define PRIMALSOLVER_H_
#include <iostream>
#if (defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) || defined(__WINDOWS__) || defined(MINGW)) && !defined(CYGWIN)
#undef MAXSIZE_T
#endif
#include <numeric>
#include <utility>
#include <queue>
#include <algorithm>
#include <functional>
#include <stdexcept>
#include <iomanip>
#include <cassert>
#include <cmath>
#include <list>
#include <limits>

//#define TRWS_DEBUG_OUTPUT

#ifdef TRWS_DEBUG_OUTPUT
#include <opengm/inference/trws/output_debug_utils.hxx>
#endif

namespace TransportSolver
{

#ifdef TRWS_DEBUG_OUTPUT
using OUT::operator <<;
#endif

/* List 2D class and implementation ==================================================== */

template<class T>
class List2D
{
public:

	struct bufferElement;

	struct listElement
	{
		listElement(size_t coordinate,bufferElement* pbufElement):
			_coordinate(coordinate), _pbufElement(pbufElement)
		{};

		size_t _coordinate;
		bufferElement* _pbufElement;
	};

	typedef std::list<listElement> List1D;

	template<class Parent,class typeT>
		class iterator_template : public Parent
		{
			public:
			iterator_template(Parent it,const bufferElement* pbuffer0):Parent(it),_pbuffer0(pbuffer0){};
			typeT& operator * ()const{return this->Parent::operator *()._pbufElement->_value;}

			size_t index()const
			{
				return (this->Parent::operator *()._pbufElement)-_pbuffer0;
			}

			size_t coordinate()const{return this->Parent::operator *()._coordinate;}
			size_t x()const{return (*this->Parent::operator *()._pbufElement->_rowIterator)._coordinate;}
			size_t y()const{return (*this->Parent::operator *()._pbufElement->_colIterator)._coordinate;}

			bool isRowIterator()const{return &(*(this->Parent::operator *()._pbufElement->_rowIterator)) == &(*Parent(*this));}

			iterator_template changeDir()const{
			    if (isRowIterator())
					return iterator_template(this->Parent::operator *()._pbufElement->_colIterator,_pbuffer0);
				else
					return iterator_template(this->Parent::operator *()._pbufElement->_rowIterator,_pbuffer0);
			}

			iterator_template operator ++ (int){iterator_template it=*this; ++(*this); return it;}
			iterator_template& operator ++ (){Parent::operator ++(); return *this;}
			iterator_template& operator -- (){Parent::operator --(); return *this;}
			private:
			const bufferElement* _pbuffer0;
		};

		typedef iterator_template<typename List1D::iterator,T> iterator;
		typedef iterator_template<typename List1D::const_iterator,const T> const_iterator;

	typedef std::vector<List1D> List1DSeq;

	struct bufferElement
	{
		bufferElement(const T& val,typename List1D::iterator rowIterator,typename List1D::iterator colIterator)
		    :_value(val) {
		    if (_value != NaN()) {
			_rowIterator = rowIterator;
			_colIterator = colIterator;
		    }
		};

		bufferElement(const bufferElement &other)
		: _value(other._value)
		{
		 if (_value != NaN()) {
			 _rowIterator = other._rowIterator;
			 _colIterator = other._colIterator;
		 }
		}

		bufferElement & operator=(const bufferElement &other) {
			_value = other._value;
			 if (_value != NaN()) {
				 _rowIterator = other._rowIterator;
				 _colIterator = other._colIterator;
			 }
			 return *this;
		}

		T _value;
		typename List1D::iterator _rowIterator;
		typename List1D::iterator _colIterator;
	};

	typedef std::vector<bufferElement> Buffer;

	List2D(size_t xsize, size_t ysize, size_t nnz);
	List2D(const List2D&);
	List2D& operator = (const List2D&);

	void clear();
	/*
	 * after resizing all data is lost!
	 */
	void resize(size_t xsize, size_t ysize, size_t nnz);

	/* tries to insert to the end of lists.
	 * compares coordinates of the last element for this purpose
	 * it it does not work, returns <false>
	 */
	bool push(size_t x, size_t y, const T& val);

	/*
	 * tries to insert to the position of the last
	 * call of the function erase(). If it was not called yet -
	 * the last position in the allocated memory.
	 * If the insertion position is occupied,
	 * returns <false>
	 */
	bool insert(size_t x, size_t y, const T& val);
	void erase(iterator it);
	/*
	 * index - index in the _buffer array
	 */
	void erase(size_t index){erase(iterator(_buffer[index]._rowIterator,&_buffer[0]));}

	void rowErase(size_t y);
	void colErase(size_t x);

	size_t rowSize(size_t y)const{return _rowLists[y].size();};
	size_t xsize()const{return _colLists.size();}
	size_t colSize(size_t x)const{return _colLists[x].size();};
	size_t ysize()const{return _rowLists.size();}
	size_t nnz()const{return _buffer.size();}

	iterator rowBegin(size_t y){return iterator(_rowLists[y].begin(),&_buffer[0]);}
	const_iterator rowBegin(size_t y)const{return const_iterator(_rowLists[y].begin(),&_buffer[0]);}

	iterator rowEnd(size_t y){return iterator(_rowLists[y].end(),&_buffer[0]);}
	const_iterator rowEnd(size_t y)const{return const_iterator(_rowLists[y].end(),&_buffer[0]);}

	iterator colBegin(size_t x){return iterator(_colLists[x].begin(),&_buffer[0]);}
	const_iterator colBegin(size_t x)const{return const_iterator(_colLists[x].begin(),&_buffer[0]);}

	iterator colEnd(size_t x){return iterator(_colLists[x].end(),&_buffer[0]);}
	const_iterator colEnd(size_t x)const{return const_iterator(_colLists[x].end(),&_buffer[0]);}

	//iterator switchDirection(iterator it)const;
	template<class BinaryTable1D>
	T inner_product1D(const BinaryTable1D& bin)const;

	//pprecision - if non-zero contains an upper bound for the numerical precision of the returned  value
	template<class BinaryTable2D>
	T inner_product2D(const BinaryTable2D& bin, T* pprecision=0)const;

	template<class BinaryTable2D>
	void get2DTable(BinaryTable2D* pbin)const;

	T& buffer(size_t index){return _buffer[index]._value;}
	const T& buffer(size_t index)const{return _buffer[index]._value;}

	std::pair<bool,T> getValue(size_t x,size_t y)const;//!< not very efficient function. Implemented mainly for test purposes.

#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
#endif
private:
	bool _insert(size_t x, size_t y, const T& val, size_t position);
	void _copy(const List2D<T>& lst);
	static T NaN(){return std::numeric_limits<T>::max();}

	//size_t _nnz;
	size_t _insertPosition;
	size_t _pushPosition;
	List1DSeq _rowLists;
	List1DSeq _colLists;
	Buffer _buffer;
};

template<class T>
List2D<T>::List2D(size_t xsize, size_t ysize, size_t nnz):
_insertPosition(nnz-1),
_pushPosition(0),
_rowLists(ysize),
_colLists(xsize),
_buffer(nnz,bufferElement(NaN(),typename List1D::iterator(),typename List1D::iterator()))
{};

template<class T>
List2D<T>::List2D(const List2D& lst)
{
	_copy(lst);
}

template<class T>
void List2D<T>::resize(size_t xsize, size_t ysize, size_t nnz)
{
	_rowLists.assign(ysize,List1D());
	_colLists.assign(xsize,List1D());
	_buffer.assign(nnz,bufferElement(NaN(),typename List1D::iterator(),typename List1D::iterator()));
	_insertPosition=nnz-1;
	_pushPosition=0;
};


template<class T>
void List2D<T>::_copy(const List2D<T>& lst)
{
	_buffer=lst._buffer;

	_rowLists=lst._rowLists;
	typename List1DSeq::iterator itbeg=_rowLists.begin(), itend=_rowLists.end();
	for (;itbeg!=itend;++itbeg)
	{
		typename List1D::iterator beg=(*itbeg).begin(),end=(*itbeg).end();
		for (;beg!=end;++beg)
		{
			size_t offset=(*beg)._pbufElement-&(lst._buffer[0]);
			(*beg)._pbufElement= &_buffer[offset];
			(*beg)._pbufElement->_rowIterator=beg;
		}
	}

	_colLists=lst._colLists;
	itbeg=_colLists.begin(), itend=_colLists.end();
	for (;itbeg!=itend;++itbeg)
	{
		typename List1D::iterator beg=(*itbeg).begin(),end=(*itbeg).end();
		for (;beg!=end;++beg)
		{
			size_t offset=(*beg)._pbufElement-&(lst._buffer[0]);
			(*beg)._pbufElement= &_buffer[offset];
			(*beg)._pbufElement->_colIterator=beg;
		}
	}

	//_nnz=lst._nnz;
	_insertPosition=lst._insertPosition;
	_pushPosition=lst._pushPosition;
};

template<class T>
List2D<T>& List2D<T>::operator = (const List2D<T>& lst)
{
	if (this==&lst)
		return *this;

	_copy(lst);

	return *this;
}

template<class T>
bool List2D<T>::insert(size_t x,size_t y,const T& val)
{
	if (_insert(x,y,val,_insertPosition))
	{
	  _insertPosition=_buffer.size();
	  return true;
	}

	return false;
};

template<class T>
bool List2D<T>::push(size_t x, size_t y, const T& val)
{
	if (_insert(x,y,val,_pushPosition))
	{
	  ++_pushPosition;
	  //the very last position in _buffer can not be occupied de to push(), only due to insert()
	  if (_pushPosition == (_buffer.size()-1))
		  ++_pushPosition;
	  return true;
	}

	return false;
}

template<class E>
class coordLess
{
public:
 coordLess(size_t x):_x(x){}
 bool operator () (const E& e) const{return e._coordinate < _x;}
private:
 size_t _x;
};

template<class E>
class coordMore
{
public:
 coordMore(size_t x):_x(x){}
 bool operator () (const E& e) const{return e._coordinate > _x;}
private:
 size_t _x;
};

template<class T>
bool List2D<T>::_insert(size_t x, size_t y, const T& val, size_t position)
{
	assert(x<_colLists.size());
	assert(y< _rowLists.size());

	if (position >= _buffer.size())
		return false;

	bufferElement& buf=_buffer[position];
	buf._value=val;

	List1D& rowList=_rowLists[y];
	List1D& colList=_colLists[x];

	typename List1D::iterator insertPosition=std::find_if(rowList.begin(),rowList.end(),coordMore<listElement>(x));
	buf._rowIterator=rowList.insert(insertPosition,listElement(x,&buf));
	insertPosition=std::find_if(colList.begin(),colList.end(),coordMore<listElement>(y));
	buf._colIterator=colList.insert(insertPosition,listElement(y,&buf));

	return true;
};

template<class T>
void List2D<T>::erase(iterator it)
{
 _insertPosition=it.index();
 size_t x=it.x(), y=it.y();
 _rowLists[y].erase(_buffer[_insertPosition]._rowIterator);
 _colLists[x].erase(_buffer[_insertPosition]._colIterator);
 _buffer[_insertPosition]._value=NaN();
};

template<class T>
void List2D<T>::rowErase(size_t y)
{
	while (!_rowLists[y].empty())
		erase(iterator(_rowLists[y].begin(),&_buffer[0]));
};

template<class T>
void List2D<T>::colErase(size_t x)
{
	while (!_colLists[x].empty())
		erase(iterator(_colLists[x].begin(),&_buffer[0]));
};

template<class T>
void List2D<T>::clear()
{
	for (size_t x=0;x<_rowLists.size();++x)
		rowErase(x);

	for (size_t y=0;y<_colLists.size();++y)
		colErase(y);

	_pushPosition=0;
	_insertPosition=_buffer.size()-1;
};

template<class T>
template<class BinaryTable1D>
T List2D<T>::inner_product1D(const BinaryTable1D& bin)const
{
	T sum=0;
	for (size_t i=0; i<_colLists.size();++i)
	{
		typename List1D::const_iterator beg=_colLists[i].begin(), end=_colLists[i].end();
		for (;beg!=end;++beg)
			sum+=(*beg)._pbufElement->_value * bin[xsize()*((*beg)._coordinate)+i];
	};
	return sum;
};

template<class T>
template<class BinaryTable2D>
T List2D<T>::inner_product2D(const BinaryTable2D& bin, T* pprecision)const //DEBUG
{
	T floatTypeEps=std::numeric_limits<T>::epsilon();
	T precision_;
	T* pprecision_;
	if (pprecision!=0)
		pprecision_=pprecision;
	else
		pprecision_=&precision_;

	*pprecision_=0;

	T sum=0;
	for (size_t i=0; i<xsize();++i)
	{
		const_iterator beg=colBegin(i), end=colEnd(i);
		for (;beg!=end;++beg)
		{
			sum+=(*beg) * bin(beg.x(),beg.y());
			*pprecision_+=floatTypeEps*fabs(sum);
		}
	};
	return sum;
};

template<class T>
std::pair<bool,T> List2D<T>::getValue(size_t x,size_t y)const
{
 typename List1D::const_iterator beg=_colLists[x].begin(), end=_colLists[x].end();
 for (;beg!=end;++beg)
	if ((*beg)._coordinate==y)
		return std::make_pair(true,(*beg)._pbufElement->_value);

  return std::make_pair(false,(T)0);
};

#ifdef TRWS_DEBUG_OUTPUT
template<class T>
void List2D<T>::PrintTestData(std::ostream& fout)const
{
	fout << "_nnz=" <<_buffer.size()<<std::endl;
	fout << "_insertPosition=" << _insertPosition<<std::endl;
	fout << "_pushPosition=" << _pushPosition<<std::endl;
	fout << "xsize="<<_colLists.size()<<std::endl;
	fout << "ysize="<<_rowLists.size()<<std::endl;

	std::vector<T> printBuffer(_buffer.size(),NaN());

	fout << "row Lists: "<<std::endl;
	for (size_t i=0; i< _rowLists.size();++i)
	{
		fout << "y="<<i<<": ";
		typename List1D::const_iterator beg=_rowLists[i].begin(), end=_rowLists[i].end();
		for (;beg!=end;++beg)
		{
			fout <<"("<<(*beg)._coordinate<<","<<(*beg)._pbufElement->_value<<")";
			printBuffer[(*beg)._pbufElement-&_buffer[0]]=(*beg)._pbufElement->_value;
		}
		fout <<std::endl;
	}

	fout << "column Lists: "<<std::endl;
	for (size_t i=0; i< _colLists.size();++i)
	{
		fout << "x="<<i<<": ";
		typename List1D::const_iterator beg=_colLists[i].begin(), end=_colLists[i].end();
		for (;beg!=end;++beg)
		{
			fout <<"("<<(*beg)._coordinate<<","<<(*beg)._pbufElement->_value<<")";
		}
		fout <<std::endl;
	}

	fout << "buffer: ";
	for (size_t i=0;i<printBuffer.size();++i)
		if (printBuffer[i]!=NaN())
		 fout << "("<<_buffer[i]._value<<","<<(*_buffer[i]._rowIterator)._coordinate <<","<< (*_buffer[i]._colIterator)._coordinate<<")";
		else
			fout << "(nan,nan,nan)";
	fout << std::endl;

};
#endif

template<class T>
template<class BinaryTable2D>
void List2D<T>::get2DTable(BinaryTable2D* pbin)const
{
	for (size_t x=0;x<xsize();++x)
		for (size_t y=0;y<ysize();++y)
			(*pbin)(x,y)=0;

	for (size_t i=0; i<xsize();++i)
	{
		const_iterator beg=colBegin(i), end=colEnd(i);
		for (;beg!=end;++beg)
			(*pbin)(beg.x(),beg.y())=(*beg);
	};
};

//=====================================================================================

/*
 * simple matrix class
 */
template<class T>
class MatrixWrapper
{
public:
  typedef typename std::vector<T>::const_iterator const_iterator;
  typedef typename std::vector<T>::iterator iterator;
  typedef T ValueType;
  MatrixWrapper():_xsize(0),_ysize(0){};
  MatrixWrapper(size_t xsize,size_t ysize):_xsize(xsize),_ysize(ysize),_array(xsize*ysize){};
  MatrixWrapper(size_t xsize,size_t ysize, T value):_xsize(xsize),_ysize(ysize),_array(xsize*ysize,value){};
  void resize(size_t xsize,size_t ysize){_xsize=xsize;_ysize=ysize;_array.resize(xsize*ysize);}; //<! array entries will not be copied!
  void assign(size_t xsize,size_t ysize,T value){_xsize=xsize;_ysize=ysize;_array.assign(xsize*ysize,value);};
  const_iterator begin()const {return _array.begin();}
  const_iterator end  ()const {return _array.end();}
  iterator 		 begin()	  {return _array.begin();}
  iterator 		 end  ()	  {return _array.end();}

  const T& operator() (size_t x,size_t y)const{return _array[y*_xsize + x];}
  	    T& operator() (size_t x,size_t y)	  {return _array[y*_xsize + x];}
  size_t xsize()const{return _xsize;}
  size_t ysize()const{return _ysize;}

#ifdef TRWS_DEBUG_OUTPUT
  std::ostream& print(std::ostream& out)const
  {
	const_iterator it=begin();
	out<<"["<<_xsize<<","<<_ysize<<"](";
	for (size_t y=0;y<_ysize;++y)
	{
	 if (y!=0) out << ",";
	 out <<"(";
	 for (size_t x=0;x<_xsize;++x)
	 {
	  if (x!=0) out << ",";
	  out << *it;
	  ++it;
	 }
	 out << ")";
	}

	return out<<")";
  }
#endif

private:
  size_t _xsize, _ysize;
  std::vector<T> _array;
};

template<class T>
void transpose(const MatrixWrapper<T>& input, MatrixWrapper<T>& result)
{
 result.resize(input.xsize(),input.ysize());
 for (size_t x=0;x<input.xsize();++x)
  for (size_t y=0;y<input.ysize();++y)
	  result(y,x)=input(x,y);
}

/*
 * Additionally to the functionality provided by std::copy_if it returns indexes of elements satisfying pred
 */
template <class InputIterator, class OutputIteratorValue,class OutputIteratorIndex, class UnaryPredicate>
OutputIteratorValue copy_if (InputIterator first, InputIterator last,
		  OutputIteratorValue result, OutputIteratorIndex resultIndex, UnaryPredicate pred)
{
  size_t indx=0;
  while (first!=last) {
    if (pred(*first)) {
      *result = *first;
      *resultIndex=indx;
      ++resultIndex;
      ++result;
    }
    ++indx;
    ++first;
  }
  return result;
}

template<class Iterator,class T>
T _Normalize(Iterator begin,Iterator end,T initialValue)
{
	T acc=std::accumulate(begin,end,(T)0.0);
	std::transform(begin,end,begin,std::bind1st(std::multiplies<T>(),1.0/acc));
	return initialValue+acc;
};

//===== TransportationSolver class ==============================================
/*
* in class OPTIMIZER the member bool bop(const T& a, const T& b) has to be defined. If minimization is meant, then bop== operator <()
* if maximization -> bop == operator >()
* OPTIMIZER == ACC in opengm notation
*
* DenseMatrix represents a dense matrix type and has to
*  - contain elements of the type floatType, defined in common.h
*  and provide floatType operator ()(size_t index_a, size_t index_b) to access its elements
*  Examples for Matrix:
*  MatrixWrapper defined in simpleobjects.h
*  boost::numeric::ublas::matrix<DD::floatType>;
*
*   see also tests/testcommon.h
**
**/
template<class OPTIMIZER, class DenseMatrix>
class TransportationSolver
{
public:
	typedef typename DenseMatrix::ValueType floatType;
	typedef enum{X, Y} Direction;
	typedef std::pair<size_t,Direction> CoordDir;
	typedef std::queue<CoordDir> Queue;
	typedef List2D<floatType> FeasiblePoint;
	typedef std::vector<floatType> UnaryDense;
	typedef std::vector<size_t> IndexArray;
	typedef std::list<typename FeasiblePoint::const_iterator> CycleList;

	static const floatType floatTypeEps;
	static const size_t defaultMaxIterationNumber;
	static const size_t MAXSIZE_T;

	TransportationSolver(
#ifdef	TRWS_DEBUG_OUTPUT
			std::ostream& fout=std::cerr,
#endif
			floatType relativePrecision=floatTypeEps,size_t maxIterationNumber=defaultMaxIterationNumber):
#ifdef	TRWS_DEBUG_OUTPUT
		_fout(fout),
#endif
		_pbinInitial(0),_xsize(0),_ysize(0),_relativePrecision(relativePrecision),_basicSolution(0,0,0),_maxIterationNumber(maxIterationNumber)
	{
		assert(relativePrecision >0);
	};

	TransportationSolver(const size_t& xsize,const size_t& ysize,const DenseMatrix& bin,
#ifdef	TRWS_DEBUG_OUTPUT
			std::ostream& fout=std::cout,
#endif
			floatType relativePrecision=floatTypeEps,size_t maxIterationNumber=100):
#ifdef	TRWS_DEBUG_OUTPUT
		_fout(fout),
#endif
	  _pbinInitial(&bin),_xsize(xsize),_ysize(ysize),_relativePrecision(relativePrecision),_basicSolution(xsize,ysize,_nnz(xsize,ysize)),_maxIterationNumber(maxIterationNumber)
	{
		assert(relativePrecision >0);
		Init(xsize,ysize,bin);
	};


	void Init(size_t xsize,size_t ysize,const DenseMatrix& bin);
	/*
	 * iterators xbegin and ybegin should point out to containers at least xsize and ysize long.
	 * Only the first xsize and ysize will be used.
	 * Iterator should support operation + n, i.e. begin+xsize should be defined
	 * Non-necessary near-zero elements will NOT be considered automatically to avoid numerical problems and save computational time
	 */
	template <class Iterator>
	floatType Solve(Iterator xbegin,Iterator ybegin);

	floatType GetObjectiveValue()const{return _basicSolution.inner_product2D(_matrix, &_primalValueNumericalPrecision);};//!< returns value of the current basic solution
	/*
	 * OutputMatrix should provide operator ()(size_t index_a, size_t index_b) to assign values to its elements
	 */
	template<class OutputMatrix>
	floatType GetSolution(OutputMatrix* pbin)const;
#ifdef TRWS_DEBUG_OUTPUT
	void PrintTestData(std::ostream& fout)const;
	void PrintProblemDescription(const UnaryDense& xarr,const UnaryDense& yarr);
#endif
private:
	void _InitBasicSolution(const UnaryDense& xarr,const UnaryDense& yarr);
	bool _isOptimal(std::pair<size_t,size_t>* pmove);
	bool _CheckDualConstraints(const UnaryDense& xdual,const UnaryDense& ydual,std::pair<size_t,size_t>* pmove )const;
	CoordDir _findSingleNeighborNode(const FeasiblePoint&)const;
	void _BuildDuals(UnaryDense* pxdual,UnaryDense* pydual);
	void _FindCycle(FeasiblePoint* pfp,const std::pair<size_t,size_t>& move);
	void _ChangeSolution(const FeasiblePoint& fp,const std::pair<size_t,size_t>& move);
	bool _MovePotentials(const std::pair<size_t,size_t>& move);

	void _move2Neighbor(const FeasiblePoint& fp,typename FeasiblePoint::const_iterator &it)const;//helper function - determines, iterator direction and moves it to another element of a list fp with length 2

	static size_t _nnz(size_t xsize,size_t ysize){return xsize+ysize;}
	template <class Iterator>
	static floatType _FilterBound(Iterator xbegin,size_t xsize,UnaryDense& out,IndexArray* pactiveIndexes, floatType precision);
	void _FilterObjectiveMatrix();
	void _checkCounter(size_t* pcounter,const char* errmess);

	mutable floatType _primalValueNumericalPrecision;
	bool _recalculated;
#ifdef TRWS_DEBUG_OUTPUT
	std::ostream& _fout;
#endif
	const DenseMatrix* _pbinInitial;
	MatrixWrapper<floatType> _matrix;
	size_t _xsize,_ysize;
	floatType _relativePrecision;//relative precision of thresholding input marginal values
	FeasiblePoint _basicSolution;
	IndexArray _nonZeroXcoordinates;//_activeXbound;
	IndexArray _nonZeroYcoordinates;//_activeYbound;
	size_t _maxIterationNumber;
};

template<class OPTIMIZER,class DenseMatrix>
const typename TransportationSolver<OPTIMIZER,DenseMatrix>::floatType TransportationSolver<OPTIMIZER,DenseMatrix>::floatTypeEps=std::numeric_limits<TransportationSolver<OPTIMIZER,DenseMatrix>::floatType>::epsilon();

template<class OPTIMIZER,class DenseMatrix>
const size_t TransportationSolver<OPTIMIZER,DenseMatrix>::MAXSIZE_T=std::numeric_limits<size_t>::max();

template<class OPTIMIZER,class DenseMatrix>
const size_t TransportationSolver<OPTIMIZER,DenseMatrix>::defaultMaxIterationNumber=100;

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::Init(size_t xsize,size_t ysize,const DenseMatrix& bin)
{
	_pbinInitial=&bin;
	_xsize=xsize;
	_ysize=ysize;
	_basicSolution.resize(xsize,ysize,_nnz(xsize,ysize));
	_nonZeroXcoordinates.clear();
	_nonZeroYcoordinates.clear();
	_primalValueNumericalPrecision=0;
};

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_checkCounter (size_t* pcounter, const char* errmess)
{
	if ((*pcounter)++ < std::max(_xsize*_ysize*100,_maxIterationNumber) )//100 - magic number
		return;

	throw std::runtime_error(errmess);
};

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_InitBasicSolution(const UnaryDense& xarr,const UnaryDense& yarr)
{
	UnaryDense row=xarr;
	UnaryDense col=yarr;
	//north-west corner basic solution
	//_basicSolution.clear();
	_basicSolution.resize(xarr.size(),yarr.size(),_nnz(xarr.size(),yarr.size()));
	typename UnaryDense::iterator rbeg=row.begin(), rend=row.end();
	typename UnaryDense::iterator cbeg=col.begin(), cend=col.end();

	size_t counter=0;
	while ((rbeg!=rend)&&(cbeg!=cend))
	{
	if (*cbeg>=*rbeg)
	{
		//_basicSolution.push(rbeg.index(), cbeg.index(),*rbeg);
		_basicSolution.push(rbeg-row.begin(), cbeg-col.begin(),*rbeg);
		(*cbeg)-=(*rbeg);
		if (rbeg!=rend)
		 ++rbeg;
		else
		 ++cbeg;
	}
	else
	{
		_basicSolution.push(rbeg-row.begin(),cbeg-col.begin(),*cbeg);
		(*rbeg)-=(*cbeg);
		if (cbeg!=cend)
		 ++cbeg;
		else
		 ++rbeg;
	}

	_checkCounter(&counter,"_InitBasicSolution-infinite loop!\n");
	}

	size_t basicNum=xarr.size()+yarr.size()-1;
	if (counter!=basicNum)
		throw std::runtime_error("TransportationSolver::_InitBasicSolution() : INTERNAL ERROR: Can not initialize basic solution!");
};


/*
 * returns coordinate + direction of the point which is alone in its row/column.
 * e.g. (1,X) means that the column with X-coordinate equal to 1, contains a single element.
 */

template<class OPTIMIZER,class DenseMatrix>
typename TransportationSolver<OPTIMIZER,DenseMatrix>::CoordDir
TransportationSolver<OPTIMIZER,DenseMatrix>::_findSingleNeighborNode(const FeasiblePoint& fp)const
{
	for (size_t i=0;i<_nonZeroXcoordinates.size();++i)
		if (fp.colSize(i)==1)
			return std::make_pair(i,X);

	for (size_t i=0;i<_nonZeroYcoordinates.size();++i)
		if (fp.rowSize(i)==1)
			return std::make_pair(i,Y);

	return std::make_pair(MAXSIZE_T,X);
};

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_BuildDuals(UnaryDense* pxdual,UnaryDense* pydual)
{
	UnaryDense& xdual=*pxdual;
	UnaryDense& ydual=*pydual;

	xdual.assign(_nonZeroXcoordinates.size(),0.0);
	ydual.assign(_nonZeroYcoordinates.size(),0.0);

	FeasiblePoint fpcopy(_basicSolution);
	CoordDir currNode=_findSingleNeighborNode(fpcopy);

	if (currNode.first==MAXSIZE_T)
		throw std::runtime_error("_BuildDuals: can not build duals: no single neighbor node available!");

	if (currNode.second==X)
	{
		currNode.second=Y;
		currNode.first=fpcopy.colBegin(currNode.first).coordinate();
	}else
	{
		currNode.second=X;
		currNode.first=fpcopy.rowBegin(currNode.first).coordinate();
	}

	Queue qu;
	qu.push(currNode);

	size_t counter=0;
	do
	{
		if (qu.front().second==Y)
		{
		 size_t y=qu.front().first;

		 typename FeasiblePoint::iterator beg=fpcopy.rowBegin(y),
										  end=fpcopy.rowEnd(y);
		 for (;beg!=end;++beg)
		 {
			 size_t x=beg.coordinate();
			 //xdual[x]=(*_pbin)(x,y)-ydual[y];
			 xdual[x]=_matrix(x,y)-ydual[y];
			 qu.push(std::make_pair(x,X));
		 }
		 fpcopy.rowErase(y);

		}else
		{
			size_t x=qu.front().first;

			 typename FeasiblePoint::iterator beg=fpcopy.colBegin(x),
											  end=fpcopy.colEnd(x);
			 for (;beg!=end;++beg)
			 {
				 size_t y=beg.coordinate();
				 //ydual[y]=(*_pbin)(x,y)-xdual[x];
				 ydual[y]=_matrix(x,y)-xdual[x];
				 qu.push(std::make_pair(y,Y));
			 }

			 fpcopy.colErase(x);
		}

		qu.pop();

	_checkCounter(&counter, "_BuildDuals-infinite loop!\n");
	}while (!qu.empty());

};

template<class OPTIMIZER,class DenseMatrix>
 bool TransportationSolver<OPTIMIZER,DenseMatrix>::
_CheckDualConstraints(const UnaryDense& xdual,const UnaryDense& ydual,std::pair<size_t,size_t>* pmove)const
{
	floatType eps=(OPTIMIZER::bop(1,0) ? 1.0 : -1.0)*floatTypeEps;
	floatType delta, precision;

	typename MatrixWrapper<floatType>::const_iterator mit=_matrix.begin();
	for (typename UnaryDense::const_iterator ybeg=ydual.begin();ybeg<ydual.end();++ybeg)
		for (typename UnaryDense::const_iterator xbeg=xdual.begin();xbeg<xdual.end();++xbeg)
		{
			delta=*mit-*xbeg-*ybeg;
			precision=(fabs(*mit)+fabs(*xbeg)+fabs(*ybeg))*eps;
			if (OPTIMIZER::bop(delta,precision))
			 {
				pmove->first=xbeg-xdual.begin(); pmove->second=ybeg-ydual.begin();
				return false;
			 }
			++mit;
		}

		return true;
};

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_FindCycle(FeasiblePoint* pfp,const std::pair<size_t,size_t>& move)
{
	FeasiblePoint& fp=*pfp;

	fp.insert(move.first,move.second,0);

	CoordDir cd=_findSingleNeighborNode(fp);

	size_t counter=0;//_initCounter();
	while (cd.first<MAXSIZE_T)
	{
		if (cd.second==X)
			fp.colErase(cd.first);
		else
			fp.rowErase(cd.first);

		cd=_findSingleNeighborNode(fp);

		_checkCounter(&counter,"_FindCycle-infinite loop!\n");
	}
};

//helper function - determines, iterator direction and moves it to another element of a list fp with length 2
template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_move2Neighbor(const FeasiblePoint& fp,typename FeasiblePoint::const_iterator &it)const
{
 typename FeasiblePoint::const_iterator beg=fp.rowBegin(0);
 if (it.isRowIterator())
 {
	 beg=fp.rowBegin(it.y());
 }
 else
 {
	 beg=fp.colBegin(it.x());
 }

if (beg==it)
	++it;
else
	--it;
};

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
_ChangeSolution(const FeasiblePoint& fp,const std::pair<size_t,size_t>& move)
{
	size_t y=0;
	for (;y<fp.ysize();++y)
	{
	  assert( (fp.rowSize(y)==2) || (fp.rowSize(y)==0) ) ;
	  if (fp.rowSize(y)!=0)
		break;
	}

	CycleList plusList, minusList;
	CycleList* pplus=&plusList, *pPlusList=0;
	CycleList* pminus=&minusList, *pMinusList=0;

	//going along the cycle to assign +/- to vertices correctly
	typename FeasiblePoint::const_iterator it=fp.rowBegin(y);
	std::pair<size_t,size_t> c0(it.x(),it.y());
	do{
		pplus->push_back(it);
		if ( (it.x()==move.first) && (it.y()==move.second) )
		{
			pMinusList=pminus; //really minus list is in *pMinusList
			pPlusList =pplus;
		}
		it=it.changeDir();
		_move2Neighbor(fp,it);
		swap(pplus,pminus);
	}while (! ( (it.x()==c0.first) && (it.y()==c0.second) ) );

	assert (pMinusList!=0);
	assert (pPlusList!=0);

	//selecting the smallest number to add...
		 floatType min=std::numeric_limits<floatType>::max();
		 typename CycleList::const_iterator iterMinVal, beg=pMinusList->begin(), end=pMinusList->end();
		 for (;beg!=end;++beg)
		 {
			 if ( (  **beg < min) ||
				  ( (**beg == min) &&
					    ( (beg->y() < iterMinVal->y()) ||
					    ( ((beg->y() == iterMinVal->y())
					    		 && (beg->x() < iterMinVal->x())) ) ) 	) )
			 {
				 min=**beg;
				 iterMinVal=beg;
			 }
		 }

		  //changing
		 _basicSolution.insert(move.first,move.second,0);

		  beg=pMinusList->begin(), end=pMinusList->end();
		  for (;beg!=end;++beg)
		 	 _basicSolution.buffer((*beg).index())-=min;

		  beg=pPlusList->begin(), end=pPlusList->end();
		  for (;beg!=end;++beg)
		 	 _basicSolution.buffer((*beg).index())+=min;

		  _basicSolution.erase((*iterMinVal).index());
}

template<class OPTIMIZER,class DenseMatrix>
bool TransportationSolver<OPTIMIZER,DenseMatrix>::
_MovePotentials(const std::pair<size_t,size_t>& move)
{
	floatType ObjVal=GetObjectiveValue();
	floatType primalValueNumericalPrecisionOld=_primalValueNumericalPrecision;
	FeasiblePoint fp=_basicSolution;

	_FindCycle(&fp,move);

	_ChangeSolution(fp,move);


        floatType newObjValue=GetObjectiveValue();
	if ( (OPTIMIZER::bop(ObjVal,newObjValue)) && (fabs(ObjVal-newObjValue)>(_primalValueNumericalPrecision+primalValueNumericalPrecisionOld) ) )
		{
#ifdef	TRWS_DEBUG_OUTPUT
			std::cerr<<_fout<<std::setprecision (std::numeric_limits<floatType>::digits10+1) << std::endl<<"ObjVal="<<ObjVal
					 <<", newObjValue="<<newObjValue
					 <<", fabs(ObjVal-newObjValue)="<<fabs(ObjVal-newObjValue)<<", _primalValueNumericalPrecision="<<_primalValueNumericalPrecision
					 << ", primalValueNumericalPrecisionOld="<< primalValueNumericalPrecisionOld <<std::endl;
			_fout << "Basic solution before move:" <<std::endl;
			fp.PrintTestData(_fout);
			_fout << "Move:" << move<<std::endl;
#endif
			return false;
		}

	return true;
};


template<class OPTIMIZER,class DenseMatrix>
bool TransportationSolver<OPTIMIZER,DenseMatrix>::
_isOptimal(std::pair<size_t,size_t>* pmove)
{
	//checks current basic solution for optimality
	//1. build duals
	UnaryDense xduals,yduals;
	_BuildDuals(&xduals,&yduals);
	//2. check whether they satisfy dual constraints
	return _CheckDualConstraints(xduals,yduals,pmove);
};

template<class OPTIMIZER,class DenseMatrix>
template <class Iterator>
typename TransportationSolver<OPTIMIZER,DenseMatrix>::floatType TransportationSolver<OPTIMIZER,DenseMatrix>::
_FilterBound(Iterator xbegin,size_t xsize,UnaryDense& out,IndexArray* pactiveIndexes,floatType precision)
{
	size_t numOfMeaningfulValues=std::count_if(xbegin,xbegin+xsize,std::bind2nd(std::greater<floatType>(),precision));

 	if (numOfMeaningfulValues==0)
 		throw std::runtime_error("TransportationSolver:_FilterBound(): Error: empty output array. Was the _relativePrecision parameter selected properly?");

 	out.resize(numOfMeaningfulValues);
 	pactiveIndexes->resize(numOfMeaningfulValues);
 	TransportSolver::copy_if(xbegin,xbegin+xsize,out.begin(),pactiveIndexes->begin(),std::bind2nd(std::greater<floatType>(),precision));
 	return _Normalize(out.begin(),out.end(),(floatType)0.0);
}

#ifdef TRWS_DEBUG_OUTPUT
template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::
PrintProblemDescription(const UnaryDense& xarr,const UnaryDense& yarr)
{
		 size_t maxprecision=std::numeric_limits<floatType>::digits10;
				_fout<< std::setprecision (maxprecision+1) << "xarr=" << xarr<< std::endl;;
				_fout<< std::setprecision (maxprecision+1) << "yarr=" << yarr << std::endl;

				for (size_t x=0;x<xarr.size();++x)
				 for (size_t y=0;y<yarr.size();++y)
					_fout << std::setprecision (maxprecision+1)<<"; bin("<<_nonZeroXcoordinates[x]<<","<<_nonZeroYcoordinates[y]<<")="<<_matrix(x,y)<<std::endl;

		_fout <<std::endl<< "Current basic solution:"<<std::endl;
		_basicSolution.PrintTestData(_fout);
}
#endif

template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::_FilterObjectiveMatrix()
{
	_matrix.resize(_nonZeroXcoordinates.size(),_nonZeroYcoordinates.size());
	typename MatrixWrapper<floatType>::iterator begin=_matrix.begin();
	for (size_t y=0;y<_nonZeroYcoordinates.size();++y)
	  for (size_t x=0;x<_nonZeroXcoordinates.size();++x)
		{
			size_t ycurr=_nonZeroYcoordinates[y];
			*begin=(*_pbinInitial)(_nonZeroXcoordinates[x],ycurr);
			++begin;
		}
}

template<class OPTIMIZER,class DenseMatrix>
template<class Iterator>
typename TransportationSolver<OPTIMIZER,DenseMatrix>::floatType TransportationSolver<OPTIMIZER,DenseMatrix>::
Solve(Iterator xbegin,Iterator ybegin)
{
	_recalculated=false;
	UnaryDense xarr,yarr;

	_FilterBound(xbegin,_xsize,xarr,&_nonZeroXcoordinates,_relativePrecision*_xsize*_ysize);
	_FilterBound(ybegin,_ysize,yarr,&_nonZeroYcoordinates,_relativePrecision*_xsize*_ysize);
	_FilterObjectiveMatrix();

	//1. Create basic solution _basicSolution
	_InitBasicSolution(xarr,yarr);

	//2. Check optimality
	std::pair<size_t,size_t> move;
	bool objectiveImprovementFlag=true;
	size_t counter=0;//_initCounter();

	while ((objectiveImprovementFlag)&&(!_isOptimal(&move)))
	{
		objectiveImprovementFlag=_MovePotentials(move); //changes basic solution
		//_checkCounter(&counter,"TransportationSolver::Solve(): maximal number of iterations reached! Try to increase <maxIterationNumber> in constructor.\n");
 		if (counter++ > std::max(_xsize*_ysize*100,_maxIterationNumber))
 		{
 #ifdef TRWS_DEBUG_OUTPUT
 			_fout << "Warning! TransportationSolver::Solve(): maximal number of iterations reached! A non-optimal solution is possible!"<<std::endl;
 #endif
                        break;
 		}
		if (!objectiveImprovementFlag)
		{
#ifdef TRWS_DEBUG_OUTPUT
			PrintProblemDescription(xarr,yarr);
#endif
			throw std::runtime_error("TransportationSolver::Solve: INTERNAL ERROR: Objective has become worse. Interrupting!");
		}
	}
	return  GetObjectiveValue();
};


template<class OPTIMIZER,class DenseMatrix>
template<class OutputMatrix>
typename TransportationSolver<OPTIMIZER,DenseMatrix>::floatType TransportationSolver<OPTIMIZER,DenseMatrix>::GetSolution(OutputMatrix* pbin)const
{
	for (size_t y=0;y<_ysize;++y)
	 for (size_t x=0;x<_xsize;++x)
		 (*pbin)(x,y)=0.0;

	MatrixWrapper<floatType> matrix(_basicSolution.xsize(),_basicSolution.ysize());
	_basicSolution.get2DTable(&matrix);
	for (size_t y=0;y<matrix.ysize();++y)
	 for (size_t x=0;x<matrix.xsize();++x)
		 (*pbin)(_nonZeroXcoordinates[x],_nonZeroYcoordinates[y])=matrix(x,y);

	return GetObjectiveValue();
};

#ifdef TRWS_DEBUG_OUTPUT
template<class OPTIMIZER,class DenseMatrix>
void TransportationSolver<OPTIMIZER,DenseMatrix>::PrintTestData(std::ostream& fout)const
{
	fout << "_relativePrecision="<<_relativePrecision<<std::endl;
	fout << "_xsize="<<_xsize<<", _ysize="<<_ysize<<std::endl;
	fout <<"_basicSolution:"<<std::endl;
	_basicSolution.PrintTestData(fout);
	fout <<std::endl<< "_nonZeroXcoordinates: "<<_nonZeroXcoordinates;
	fout <<std::endl<< "_nonZeroYcoordinates: "<<_nonZeroYcoordinates;
};
#endif

};//TS

#endif
