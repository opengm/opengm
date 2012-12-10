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

template <typename T>
inline PyArray_TYPES typeEnumFromType(void);

using namespace boost::python;

template<class T, size_t DIM>
class NumpyViewAccessor1d;

template<class V,size_t DIM=0>
class NumpyView{
public:
   typedef V ValueType;
   typedef V  * CastPtrType;
   typedef int const * ShapePtrType;
   typedef typename marray::View< V ,false >::const_iterator IteratorType;
   
   typedef size_t const *  ShapeIteratorType;
   
   NumpyView( ):allocFromCpp_(false){
      
   }
   NumpyView( boost::python::numeric::array  array):allocFromCpp_(false){
      
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
      view_.assign(myshape.begin(),myshape.end(),mystrides.begin(),dataPtr,marray::FirstMajorOrder);
      //std::cout<<" done construct array \n";
   }
   template<class ITERATOR>
   NumpyView(ITERATOR shapeBegin,ITERATOR shapeEnd):allocFromCpp_(true){
      opengm::FastSequence<int> intShape;
      intShape.assign(shapeBegin,shapeEnd);
      // allocate array
      intp n = size;
      obj_ = boost::python::object(boost::python::handle<>(PyArray_FromDims(int(intShape.size()), intShape.begin(), typeEnumFromType<ValueType > ())));
      arrayData_ = PyArray_DATA((PyArrayObject*) obj_.ptr());
      //const boost::python::tuple strides = boost::python::extract<boost::python::tuple > (arrayData_.attr("strides"));
      ValueType * castPtr = static_cast< ValueType *>(arrayData_);
      view_.assign(intShape.begin(),intShape.end(),castPtr,marray::FirstMajorOrder,marray::FirstMajorOrder);
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
   template<class ITERATOR>
   ValueType & operator[](ITERATOR  iterator){
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

   IteratorType begin()const{ 
      return view_.begin();
   }
   IteratorType end()const{ 
      return view_.end();
   }
   IteratorType begin(){ 
      return view_.begin();
   }
   IteratorType end(){ 
      return view_.end();   
   }

   boost::python::object object()const{
      return obj_;
   };
private:
   marray::View< V ,false > view_;
   boost::python::object obj_;
   bool allocFromCpp_;
   void * arrayData_;
};




#endif	/* NUMPYVIEW_HXX */

