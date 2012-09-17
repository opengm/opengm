/* 
 * File:   numpyview.hxx
 * Author: tbeier
 *
 * Created on August 26, 2012, 4:22 PM
 */

#ifndef NUMPYVIEW_HXX
#define	NUMPYVIEW_HXX

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/noprefix.h>
#ifdef Bool
#undef Bool
#endif 
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

using namespace boost::python;

template<class T, size_t DIM>
class NumpyViewAccessor1d;

template<class V,size_t DIM=0>
class NumpyView{
public:
   typedef V ValueType;
   typedef V const * CastPtrType;
   typedef int const * ShapePtrType;
   typedef typename marray::View< V ,true >::const_iterator IteratorType;
   NumpyView( boost::python::numeric::array  array){
      void * voidDataPtr=PyArray_DATA(array.ptr());
      CastPtrType dataPtr = static_cast<CastPtrType>(voidDataPtr);
      // shape and strides extraction
      const boost::python::tuple shape = boost::python::extract<boost::python::tuple > (array.attr("shape"));
      const boost::python::tuple strides = boost::python::extract<boost::python::tuple > (array.attr("strides"));
      size_t dimension = boost::python::len(shape);
      opengm::FastSequence<size_t> myshape(dimension);
      opengm::FastSequence<size_t> mystrides(dimension);
      //std::cout<<" get shape and strides of array \n";
      for(size_t i=0;i<dimension;++i){
         myshape[i]=boost::python::extract<int>(shape[i]);
         mystrides[i]=boost::python::extract<int>(strides[i])/sizeof(V);
      }
      //std::cout<<" construct array \n";
      view_.assign(myshape.begin(),myshape.end(),mystrides.begin(),dataPtr,marray::LastMajorOrder);
      //std::cout<<" done construct array \n";
   }
   size_t size()const {return view_.size();}
   size_t dimension()const{return view_.dimension();}
   size_t shape(const size_t i)const{return view_.shape(i);}
   const size_t * shapeBegin()const{return view_.shapeBegin();}
   const size_t * shapeEnd()const{return view_.shapeEnd();}
   void error(const std::string &reason=std::string(" "))const{throw opengm::RuntimeError(reason);}
   
   const ValueType & operator()(const size_t x0)const{
      return view_(x0);
   }
   const ValueType & operator()(const size_t x0,const size_t x1)const{
      return view_(x0,x1);
   }
   
   template<class ITERATOR>
   const ValueType & operator[](ITERATOR  iterator)const{
      return view_(iterator);
   }

   
   IteratorType begin1d()const{ 
      return view_.begin();
   }
   IteratorType end1d()const{ 
      return view_.end();
   }
   IteratorType begin1d(){ 
      return view_.begin();
   }
   IteratorType end1d(){ 
      return view_.end();   
   }
private:
   marray::View< V ,true > view_;
};




#endif	/* NUMPYVIEW_HXX */

