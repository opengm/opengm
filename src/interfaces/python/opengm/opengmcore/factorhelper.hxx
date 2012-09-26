#ifndef FACTORHELPER_HXX
#define	FACTORHELPER_HXX
   
#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "iteratorToTuple.hxx"
#include "export_typedes.hxx"
#include "utilities/shapeHolder.hxx"
#include "copyhelper.hxx"
#include "../converter.hxx"

namespace pyfactor {
   
   
   template<class FACTOR,class IFACTOR>
   IFACTOR * iFactorFromFactor
   (
      const FACTOR & factor
   ) {
      return new IFACTOR(factor);
   }
      
   
   template<class FACTOR>
   boost::python::numeric::array ifactorToNumpy
   (
   const FACTOR & factor
   ) {
      int n[1]={factor.size()};
      boost::python::object obj(boost::python::handle<>(PyArray_FromDims(1, n, typeEnumFromType<typename FACTOR::ValueType>())));
      void *array_data = PyArray_DATA((PyArrayObject*) obj.ptr());
      typename FACTOR::ValueType * castedPtr=static_cast<typename FACTOR::ValueType *>(array_data);
      opengm::ShapeWalkerSwitchedOrder<typename FACTOR::ShapeIteratorType> walker(factor.shapeBegin(),factor.numberOfVariables());
      for(size_t i=0;i<factor.size();++i,++walker)
         castedPtr[i]=factor(walker.coordinateTuple().begin());
      return boost::python::extract<boost::python::numeric::array>(obj);
   }
   
   
   template<class FACTOR,class VALUE_TYPE>
   typename FACTOR::ValueType getValuePyTuple
   (
      const FACTOR & factor, 
      boost::python::tuple labelsequence
   ) {
      typedef PythonIntTupleAccessor<VALUE_TYPE,true> Accessor;
      typedef opengm::AccessorIterator<Accessor,true> Iterator;
      Accessor accessor(labelsequence);
      Iterator begin(accessor,0);
      return factor(begin);
   }
   
   template<class FACTOR>
   typename FACTOR::ValueType getValuePyNumpy
   (
      const FACTOR  & factor, 
      NumpyView<typename FACTOR::IndexType,1> numpyView
   ) {
      return factor(numpyView.begin1d());
   }
   
   template<class FACTOR,class VALUE_TYPE>
   typename FACTOR::ValueType getValuePyList
   (
      const FACTOR & factor, 
      const boost::python::list & labelsequence
   ) {
      typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
      typedef opengm::AccessorIterator<Accessor,true> Iterator;
      Accessor accessor(labelsequence);
      Iterator begin(accessor,0);
      return factor(begin);
   }

   
   template<class FACTOR,class VALUE_TYPE>
   boost::python::tuple getShapeCallByReturnPyTuple
   (
      const FACTOR & factor
   ) {
      const size_t dimension=factor.numberOfVariables();
      return iteratorToTuple<typename FACTOR::ShapeIteratorType,VALUE_TYPE,VALUE_TYPE>(factor.shapeBegin(),dimension,1);
   }
   
   template<class FACTOR>
   FactorShapeHolder<   FACTOR> getShapeHolder
   (
      const FACTOR & factor
   ) {
      return FactorShapeHolder<  FACTOR >(factor);
   }
   template<class FACTOR>
   FactorViHolder<   FACTOR> getViHolder
   (
      const FACTOR & factor
   ) {
      return FactorViHolder<  FACTOR >(factor);
   }

   template<class FACTOR,class VALUE_TYPE>
   boost::python::tuple getVisCallByReturnPyTuple
   (
      const FACTOR & factor
   ) {
      const size_t dimension=factor.numberOfVariables();
      return iteratorToTuple<typename FACTOR::VariablesIteratorType,VALUE_TYPE,VALUE_TYPE>(factor.variableIndicesBegin(),dimension,-1);
   }

   template<class FACTOR>
   boost::python::numeric::array copyValuesCallByReturnPy
   (
   const FACTOR & factor
   ) {
      int n[1]={factor.size()};
      boost::python::object obj(boost::python::handle<>(PyArray_FromDims(1, n, typeEnumFromType<typename FACTOR::ValueType>())));
      void *array_data = PyArray_DATA((PyArrayObject*) obj.ptr());
      typename FACTOR::ValueType * castedPtr=static_cast<typename FACTOR::ValueType *>(array_data);
      factor.copyValues(castedPtr);
      return boost::python::extract<boost::python::numeric::array>(obj);
   }
   
   template<class FACTOR>
   boost::python::numeric::array copyValuesSwitchedOrderCallByReturnPy
   (
   const FACTOR & factor
   ) {
      int n[1]={factor.size()};
      boost::python::object obj(boost::python::handle<>(PyArray_FromDims(1, n, typeEnumFromType<typename FACTOR::ValueType>())));
      void *array_data = PyArray_DATA((PyArrayObject*) obj.ptr());
      typename FACTOR::ValueType * castedPtr=static_cast<typename FACTOR::ValueType *>(array_data);
      factor.copyValuesSwitchedOrder(castedPtr);
      return boost::python::extract<boost::python::numeric::array>(obj);
   }
   
   template<class FACTOR>
   std::string printFactorPy(const FACTOR & factor) {
      std::stringstream ostr;
      ostr<<"Vi=(";
      const size_t numVar=factor.numberOfVariables();
      for(size_t v=0;v<numVar;++v){
            ostr<<factor.variableIndex(v)<<",";
      }
      ostr<<") Shape=(";
      for(size_t v=0;v<numVar;++v){
            ostr<<factor.shape(v)<<",";
      }
      ostr<<")"; 
      return ostr.str();
   }

}


namespace pyacc{

   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType> *
   accSomeCopyPyList
   (
      const FACTOR & factor,
      boost::python::list accVi
   ){
      IteratorHolder< PythonIntListAccessor<INDEX_TYPE,true> > holder(accVi);
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=new IndependentFactor;
      opengm::accumulate<ACC>(factor,holder.begin(),holder.end(),*independentFactor);
      return independentFactor;
   }
   
 
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType> *
   accSomeCopyPyTuple
   (
      const FACTOR & factor,
      boost::python::tuple accVi
   ){
      IteratorHolder< PythonIntTupleAccessor<INDEX_TYPE,true> > holder(accVi);
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=new IndependentFactor;
      opengm::accumulate<ACC>(factor,holder.begin(),holder.end(),*independentFactor);
      return independentFactor;
   }
   

   
   template<class FACTOR,class ACC>
   inline opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType> *
   accSomeCopyPyNumpy
   (
      const FACTOR & factor,
      NumpyView<typename FACTOR::IndexType,1> accVi
   ){
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=new IndependentFactor;
      opengm::accumulate<ACC>(factor,accVi.begin1d(),accVi.end1d(),*independentFactor);
      return independentFactor;
   }
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline void
   accSomeIFactorInplacePyList
   (
      FACTOR & factor,
      boost::python::list accVi
   ){
      IteratorHolder< PythonIntListAccessor<INDEX_TYPE,true> > holder(accVi);
      opengm::accumulate<ACC>(factor,holder.begin(),holder.end());
   }
   
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline void
   accSomeIFactorInplacePyTuple
   (
      FACTOR & factor,
      boost::python::tuple accVi
   ){
      IteratorHolder< PythonIntTupleAccessor<INDEX_TYPE,true> > holder(accVi);
      opengm::accumulate<ACC>(factor,holder.begin(),holder.end());
   }

   template<class FACTOR,class ACC>
   inline void
   accSomeIFactorInplacePyNumpy
   (
      FACTOR & factor,
      NumpyView<typename FACTOR::IndexType,1> accVi
   ){
      opengm::accumulate<ACC>(factor,accVi.begin1d(),accVi.end1d());
   }
}

#endif	/* FACTORHELPER_HXX */

