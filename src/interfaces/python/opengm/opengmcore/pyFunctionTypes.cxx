#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCore

#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "nifty_iterator.hxx"
#include "iteratorToTuple.hxx"
#include "export_typedes.hxx"
#include "copyhelper.hxx"

#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#
using namespace boost::python;


namespace pyfunction{
   
   template<class F,class VALUE_TYPE>
   typename F::ValueType getValuePyTuple
   (
      const F & function, 
      const boost::python::tuple & labelsequence
   ) {
      IteratorHolder< PythonIntTupleAccessor<VALUE_TYPE,true> > holder(labelsequence);
      return function(holder.begin());
   }
   
   template<class F,class VALUE_TYPE>
   typename F::ValueType getValuePyList
   (
      const F & function,  
      const boost::python::list & labelsequence
   ) {
      typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
      typedef opengm::AccessorIterator<Accessor,true> Iterator;
      Accessor accessor(labelsequence);
      Iterator begin(accessor,0);
      return function(begin);
   }

   
   template<class F,class VALUE_TYPE>
   boost::python::tuple getShapeCallByReturnPyTuple
   (
      const F & function
   ) {
      const size_t dimension=function.dimension();
      typedef typename F::FunctionShapeIteratorType IteratorType;
      typedef VALUE_TYPE V;
      return  iteratorToTuple<IteratorType,V,V>(function.functionShapeBegin(),dimension,static_cast<V>(1));
   }
   
   
   
   template<class FUNCTION>
   void setValueScalar(FUNCTION & f,const size_t index,const typename FUNCTION::ValueType value){
      f(index)=value;
   }   
   

   template<class FUNCTION,class VALUE_TYPE>
   void explicitFunctionAssignPyTuple
   (
      FUNCTION & f,
      const boost::python::tuple & shape
   ) {
      typedef PythonIntTupleAccessor<VALUE_TYPE,true> Accessor;
      typedef opengm::AccessorIterator<Accessor,true> Iterator;
      Accessor accessor(shape);
      Iterator begin(accessor,0);
      Iterator end(accessor,accessor.size());
      f= FUNCTION(begin,end);
   }
   
   template<class FUNCTION,class VALUE_TYPE>
   void explicitFunctionAssignPyList
   (
      FUNCTION & f,
      const boost::python::list & shape
   ) {
      typedef PythonIntListAccessor<VALUE_TYPE,true> Accessor;
      typedef opengm::AccessorIterator<Accessor,true> Iterator;
      Accessor accessor(shape);
      Iterator begin(accessor,0);
      Iterator end(accessor,accessor.size());
      f= FUNCTION(begin,end);
   }
   
   template<class FUNCTION,class VALUE_TYPE>
   FUNCTION * explicitFunctionConstructorPyList(const boost::python::list & shape){
      IteratorHolder< PythonIntListAccessor<VALUE_TYPE,true> > holder(shape);
      return new FUNCTION(holder.begin(),holder.end());
   }
   template<class FUNCTION,class VALUE_TYPE>
   FUNCTION * explicitFunctionConstructorPyTuple(const boost::python::tuple & shape){
      IteratorHolder< PythonIntTupleAccessor<VALUE_TYPE,true> > holder(shape);
      return new FUNCTION(holder.begin(),holder.end());
   }
   

   
   template<class FUNCTION,class LABEL_VECTOR>
   typename FUNCTION::ValueType & getValueRefPy(FUNCTION & f,const LABEL_VECTOR & state){
      return f(state.begin());
   }
   template<class FUNCTION,class LABEL_VECTOR>
   const typename FUNCTION::ValueType & getValueConstRefPy(const FUNCTION & f,const LABEL_VECTOR & state){
      return f(state.begin());
   }
}

#define FUNCTION_TYPE_EXPORTER_HELPER(CLASS_NAME,CLASS_STRING)\
class_<CLASS_NAME > (CLASS_STRING, init<const CLASS_NAME &> ())\
.def(init<>())\
.add_property("size", &CLASS_NAME::size)\
.add_property("dimension", &CLASS_NAME::dimension)\
.add_property("shape",&pyfunction::getShapeCallByReturnPyTuple< CLASS_NAME,int >)\
.def("__getitem__", &pyfunction::getValuePyList<CLASS_NAME,int>, return_value_policy< return_by_value >())\
.def("__getitem__", &pyfunction::getValuePyTuple<CLASS_NAME,int>, return_value_policy< return_by_value >())\
.def("__copy__", &generic__copy__< CLASS_NAME >)\
//.def("__deepcopy__", &generic__deepcopy__< CLASS_NAME >)

template<class V,class I>
void export_functiontypes(){
   typedef V ValueType;
   typedef I IndexType;
   typedef IndexType LabelType;
   typedef opengm::ExplicitFunction<ValueType,IndexType,LabelType> PyExplicitFunction;
   typedef opengm::PottsFunction<ValueType,IndexType,LabelType> PyPottsFunction;
   
   

   FUNCTION_TYPE_EXPORTER_HELPER(PyPottsFunction,"PottsFunction");

   
   FUNCTION_TYPE_EXPORTER_HELPER(PyExplicitFunction,"ExplicitFunction")
   .def("__init__", make_constructor(&pyfunction::explicitFunctionConstructorPyList<PyExplicitFunction,int> ))
   .def("__init__", make_constructor(&pyfunction::explicitFunctionConstructorPyTuple<PyExplicitFunction,int> ))
   .def("assign",&pyfunction::explicitFunctionAssignPyTuple<PyExplicitFunction,int>)
   .def("assign",&pyfunction::explicitFunctionAssignPyList<PyExplicitFunction,int>)
   ;   

//   
//
//   
  // .
   //;
   
   //.def("__init__", make_constructor(&pyfunction::explicitFunctionConstructorPy<PyExplicitFunction,LabelVector> ))
   //.def("getShape", &PyExplicitFunction::shape)
   //.def("setValue",pyfunction::setValueScalar<PyExplicitFunction>)
   //.def("value",&pyfunction::getValueConstRefPy<PyExplicitFunction,IndexTypeNumpyVector>,return_value_policy<copy_const_reference > ())
   //.def("__getitem__", &proxy<PyExplicitFunction,float>::get, return_value_policy< return_by_value >())
   //.def("__getitem__", &std_item<PyExplicitFunction,float,IndexTypeNumpyVector>::get, return_value_policy<copy_non_const_reference>())
   //.def("__setitem__", &std_item<PyExplicitFunction,float,IndexTypeNumpyVector>::set, with_custodian_and_ward<1,2>()) 
   //.def("__getitem__", &std_item<PyExplicitFunction,float,IndexTypeNumpyVector>::getC, return_value_policy<copy_non_const_reference>())
   //.def("__setitem__", &std_item<PyExplicitFunction,float,IndexTypeNumpyVector>::setC, with_custodian_and_ward<1,2>())

}

template void export_functiontypes<GmValueType,GmIndexType>();


