#ifndef FACTORHELPER_HXX
#define	FACTORHELPER_HXX

#include <boost/python.hpp>
#include <boost/python/class.hpp>
#include <boost/python/module.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>


#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include "copyhelper.hxx"
#include "nifty_iterator.hxx"
#include "utilities/shapeHolder.hxx"
#include "../gil.hxx"

namespace pyfactor {
   
   
   template<class FACTOR,class IFACTOR>
   IFACTOR * iFactorFromFactor
   (
      const FACTOR & factor
   ) {
      return new IFACTOR(factor);
   }
      
   
   template<class FACTOR>
   boost::python::object ifactorToNumpy
   (
   const FACTOR & factor
   ) {

      typedef typename FACTOR::ValueType ValueType;
      boost::python::object obj =opengm::python::get1dArray<ValueType>(factor.size());
      ValueType * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);
      opengm::ShapeWalkerSwitchedOrder<typename FACTOR::ShapeIteratorType> walker(factor.shapeBegin(),factor.numberOfVariables());
      for(size_t i=0;i<factor.size();++i,++walker)
         castedPtr[i]=factor(walker.coordinateTuple().begin());
      return obj;
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
      opengm::python::NumpyView<typename FACTOR::IndexType,1> numpyView
   ) {
      return factor(numpyView.begin1d());
   }

   template<class FACTOR>
   typename FACTOR::ValueType getValuePyVector
   (
      const FACTOR  & factor, 
      const std::vector<typename FACTOR::IndexType> vec
   ) {
      return factor(vec.begin());
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
      return opengm::python::iteratorToTuple<typename FACTOR::ShapeIteratorType,VALUE_TYPE,VALUE_TYPE>(factor.shapeBegin(),dimension,1);
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
      return opengm::python::iteratorToTuple<typename FACTOR::VariablesIteratorType,VALUE_TYPE,VALUE_TYPE>(factor.variableIndicesBegin(),dimension,-1);
   }

   template<class FACTOR>
   boost::python::object copyValuesCallByReturnPy
   (
   const FACTOR & factor
   ) {
      typedef typename FACTOR::ValueType ValueType;
      boost::python::object obj =opengm::python::get1dArray<ValueType>(factor.size());
      ValueType * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);

      {
         releaseGIL rgil;
         factor.copyValues(castedPtr);
      }
      return obj;
   }
   
   template<class FACTOR>
   boost::python::object copyValuesSwitchedOrderCallByReturnPy
   (
   const FACTOR & factor
   ) {
      typedef typename FACTOR::ValueType ValueType;
      boost::python::object obj =opengm::python::get1dArray<ValueType>(factor.size());
      ValueType * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);
      {
         releaseGIL rgil;
         factor.copyValuesSwitchedOrder(castedPtr);
      }
      return obj;
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
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=NULL;
      {
         releaseGIL rgil;
         IteratorHolder< PythonIntListAccessor<INDEX_TYPE,true> > holder(accVi);
         independentFactor=new IndependentFactor;
         opengm::accumulate<ACC>(factor,holder.begin(),holder.end(),*independentFactor);
      }
      return independentFactor;
   }
   
 
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType> *
   accSomeCopyPyTuple
   (
      const FACTOR & factor,
      boost::python::tuple accVi
   ){
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=NULL;
      {
         releaseGIL rgil;
         IteratorHolder< PythonIntTupleAccessor<INDEX_TYPE,true> > holder(accVi);
         independentFactor=new IndependentFactor;
         opengm::accumulate<ACC>(factor,holder.begin(),holder.end(),*independentFactor);
      }
      return independentFactor;
   }
   

   
   template<class FACTOR,class ACC>
   inline opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType> *
   accSomeCopyPyNumpy
   (
      const FACTOR & factor,
      opengm::python::NumpyView<typename FACTOR::IndexType,1> accVi
   ){
      typedef opengm::IndependentFactor<typename FACTOR::ValueType,typename FACTOR::IndexType,typename FACTOR::IndexType>  IndependentFactor;
      IndependentFactor *  independentFactor=NULL;
      {
         releaseGIL rgil;
         independentFactor=new IndependentFactor;
         opengm::accumulate<ACC>(factor,accVi.begin1d(),accVi.end1d(),*independentFactor);
      }
      return independentFactor;
   }
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline void
   accSomeIFactorInplacePyList
   (
      FACTOR & factor,
      boost::python::list accVi
   ){
      {
         releaseGIL rgil;
         IteratorHolder< PythonIntListAccessor<INDEX_TYPE,true> > holder(accVi);
         opengm::accumulate<ACC>(factor,holder.begin(),holder.end());
      }
   }
   
   
   template<class FACTOR,class ACC,class INDEX_TYPE>
   inline void
   accSomeIFactorInplacePyTuple
   (
      FACTOR & factor,
      boost::python::tuple accVi
   ){
      {
         releaseGIL rgil;
         IteratorHolder< PythonIntTupleAccessor<INDEX_TYPE,true> > holder(accVi);
         opengm::accumulate<ACC>(factor,holder.begin(),holder.end());
      }
   }

   template<class FACTOR,class ACC>
   inline void
   accSomeIFactorInplacePyNumpy
   (
      FACTOR & factor,
      opengm::python::NumpyView<typename FACTOR::IndexType,1> accVi
   ){
      {
         releaseGIL rgil;
         opengm::accumulate<ACC>(factor,accVi.begin1d(),accVi.end1d());
      }
   }
}

#endif	/* FACTORHELPER_HXX */

