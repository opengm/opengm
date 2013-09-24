#ifndef NUMPYVIEW_INCL_HXX
#define	NUMPYVIEW_INCL_HXX

#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <numpy/noprefix.h>
#ifdef Bool
#undef Bool
#endif 
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>

//template <typename T>
//inline PyArray_TYPES typeEnumFromType(void);

namespace opengm{
namespace python{
   
using namespace boost::python;

//template<class T, size_t DIM>
//class NumpyViewAccessor1d;

template<class V,size_t DIM=0>
class NumpyView{
public:
   typedef V ValueType;
   typedef V  * CastPtrType;
   typedef int const * ShapePtrType;
   typedef typename marray::View< V ,false >::iterator IteratorType;
   typedef typename marray::View< V ,false >::const_iterator ConstIteratorType;
   typedef size_t const *  ShapeIteratorType;
   
   NumpyView( ):allocFromCpp_(false){
   }
   NumpyView( boost::python::object  obj):allocFromCpp_(false){
      boost::python::numeric::array array = boost::python::extract<boost::python::numeric::array > (obj);
      void * voidDataPtr=PyArray_DATA(array.ptr());
      CastPtrType dataPtr = static_cast<CastPtrType>(voidDataPtr);
      size_t dimension =static_cast<size_t>(PyArray_NDIM(array.ptr()));

      npy_intp * shapePtr = PyArray_DIMS(array.ptr());
      npy_intp * stridePtr = PyArray_STRIDES(array.ptr());
      opengm::FastSequence<size_t> mystrides(dimension);
      for(size_t i=0;i<dimension;++i){
         mystrides[i]=(stridePtr[i])/sizeof(V);
      }
      view_.assign(shapePtr,shapePtr+dimension,mystrides.begin(),dataPtr,marray::FirstMajorOrder);
   }

   
   NumpyView( boost::python::numeric::array  array):allocFromCpp_(false){
      void * voidDataPtr=PyArray_DATA(array.ptr());
      CastPtrType dataPtr = static_cast<CastPtrType>(voidDataPtr);
      size_t dimension =static_cast<size_t>(PyArray_NDIM(array.ptr()));
      npy_intp * shapePtr = PyArray_DIMS(array.ptr());
      npy_intp * stridePtr = PyArray_STRIDES(array.ptr());
      opengm::FastSequence<size_t> mystrides(dimension);
      for(size_t i=0;i<dimension;++i){
         mystrides[i]=(stridePtr[i])/sizeof(V);
      }
      view_.assign(shapePtr,shapePtr+dimension,mystrides.begin(),dataPtr,marray::FirstMajorOrder);
   }


   size_t size()const {return view_.size();}
   size_t dimension()const{return view_.dimension();}
   size_t shape(const size_t i)const{return view_.shape(i);}
    size_t const * shapeBegin()const{return view_.shapeBegin();}
    size_t const * shapeEnd()const{return view_.shapeEnd();}
   void error(const std::string &reason=std::string(" "))const{throw opengm::RuntimeError(reason);}
   //ShapeIteratorType shapeBegin()const{return view_.shapeBegin();}
   //ShapeIteratorType shapeEnd()const{return view_.shapeBegin();}
   
   template<class X0>
   const ValueType & operator()(X0 x0)const{
      return view_(x0);
   }
   const ValueType & operator()(const size_t x0,const size_t x1)const{
      return view_(x0,x1);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2)const{
      return view_(x0,x1,x2);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3)const{
      return view_(x0,x1,x2,x3);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4)const{
      return view_(x0,x1,x2,x3,x4);
   }
   /*
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5)const{
      return view_(x0,x1,x2,x3,x4,x5);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6)const{
      return view_(x0,x1,x2,x3,x4,x5,x6);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7)const{
      return view_(x0,x1,x2,x3,x4,x5,x6,x7);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7,const size_t x8)const{
      return view_(x0,x1,x2,x3,x4,x5,x6,x7,x8);
   }
   const ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7,const size_t x8,const size_t x9)const{
      return view_(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
   }
   */
   template<class ITERATOR>
   const ValueType & operator[](ITERATOR  iterator)const{
      return view_(iterator);
   }
   
   template<class X0>
   ValueType & operator()(X0 x0){
      return view_(x0);
   }
   ValueType & operator()(const size_t x0,const size_t x1){
      return view_(x0,x1);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2){
      return view_(x0,x1,x2);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3){
      return view_(x0,x1,x2,x3);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4){
      return view_(x0,x1,x2,x3,x4);
   }
   /*
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5){
      return view_(x0,x1,x2,x3,x4,x5);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6){
      return view_(x0,x1,x2,x3,x4,x5,x6);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7){
      return view_(x0,x1,x2,x3,x4,x5,x6,x7);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7,const size_t x8){
      return view_(x0,x1,x2,x3,x4,x5,x6,x7,x8);
   }
   ValueType & operator()(const size_t x0,const size_t x1,const size_t x2,const size_t x3,const size_t x4,const size_t x5,const size_t x6,const size_t x7,const size_t x8,const size_t x9){
      return view_(x0,x1,x2,x3,x4,x5,x6,x7,x8,x9);
   }
   */
   template<class ITERATOR>
   ValueType & operator[](ITERATOR  iterator){
      return view_(iterator);
   }

   ConstIteratorType begin1d()const{ 
      return view_.begin();
   }
   ConstIteratorType end1d()const{ 
      return view_.end();
   }
   IteratorType begin1d(){ 
      return view_.begin();
   }
   IteratorType end1d(){ 
      return view_.end();   
   }

   ConstIteratorType begin()const{ 
      return view_.begin();
   }
   ConstIteratorType end()const{ 
      return view_.end();
   }
   IteratorType begin(){ 
      return view_.begin();
   }
   IteratorType end(){ 
      return view_.end();   
   }

   //boost::python::object arrayObject()const{
   //   return arrayObj_;
   //};
private:
   //boost::python::numeric::array arrayObj_;
   bool allocFromCpp_;
   marray::View< V ,false > view_;
   void * arrayData_;
};

}
}


#endif	/* NUMPYVIEW_INCL_HXX */

