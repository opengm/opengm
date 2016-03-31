#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

#include <stdexcept>
#include <stddef.h>
#include <vector>
#include <map>

#include "nifty_iterator.hxx"
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>

#include "copyhelper.hxx"

#include "opengm/utilities/functors.hxx"
#include "opengm/functions/explicit_function.hxx"
#include "opengm/functions/absolute_difference.hxx"
#include "opengm/functions/potts.hxx"
#include "opengm/functions/pottsn.hxx"
#include "opengm/functions/pottsg.hxx"
#include "opengm/functions/squared_difference.hxx"
#include "opengm/functions/truncated_absolute_difference.hxx"
#include "opengm/functions/truncated_squared_difference.hxx"
#include "opengm/functions/sparsemarray.hxx"




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
   typename F::ValueType getValuePyNumpy
   (
      const F & function,  
      opengm::python::NumpyView<VALUE_TYPE,1> coordinate
   ) {
      return function(coordinate.begin());
   }

   
   template<class F,class VALUE_TYPE>
   boost::python::tuple getShapeCallByReturnPyTuple
   (
      const F & function
   ) {
      const size_t dimension=function.dimension();
      typedef typename F::FunctionShapeIteratorType IteratorType;
      typedef VALUE_TYPE V;
      return  opengm::python::iteratorToTuple<IteratorType>(function.functionShapeBegin(),dimension);
   }



   
   template<class FUNCTION>
   boost::python::object copyFunctionValuesToNumpyOrder
   (
   const FUNCTION & function
   ) {
      //int n[1]={function.size()};
      typedef typename FUNCTION::ValueType ValueType;
      boost::python::object obj=opengm::python::getArray<ValueType>(function.functionShapeBegin(),function.functionShapeEnd());
      ValueType * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);
      opengm::CopyFunctor<typename FUNCTION::ValueType * > copyFunctor(castedPtr);
      function.forAllValuesInSwitchedOrder(copyFunctor);
      return obj;
   }


   template<class FUNCTION>
   char const * asString
   (
    const FUNCTION & function
   ) {
      typedef typename FUNCTION::ValueType ValueType;
      boost::python::object obj=opengm::python::getArray<ValueType>(function.functionShapeBegin(),function.functionShapeEnd());
      ValueType * castedPtr =opengm::python::getCastedPtr<ValueType>(obj);
      opengm::CopyFunctor<typename FUNCTION::ValueType * > copyFunctor(castedPtr);
      function.forAllValuesInSwitchedOrder(copyFunctor);
      return boost::python::extract<char const *>(obj.attr("__str__"));
   }
   
   

   template<class FUNCTION>
   FUNCTION * pottsFunctionConstructor(boost::python::object shape,const typename FUNCTION::ValueType ve,const typename FUNCTION::ValueType vne){
      FUNCTION * f = NULL;
      stl_input_iterator<int> begin(shape), end;
      const int s1=*begin;
      ++begin;
      const int s2=*begin;
      ++begin;
      f = new FUNCTION(s1,s2,ve,vne);
      return f;
   }


   template<class FUNCTION>
   FUNCTION * pottsNFunctionConstructor(boost::python::object shape,const typename FUNCTION::ValueType ve,const typename FUNCTION::ValueType vne){
      FUNCTION * f = NULL;
      stl_input_iterator<int> begin(shape), end;
      f = new FUNCTION(begin,end,ve,vne);
      return f;
   }

   template<class FUNCTION>
   FUNCTION * pottsGFunctionConstructor(boost::python::object shape,boost::python::object values){
      FUNCTION * f = NULL;
      stl_input_iterator<int> beginS(shape), endS;
      stl_input_iterator<typename FUNCTION::ValueType> beginV(values), endV;

      if(std::distance(beginV,endV)!=0)
         f = new FUNCTION(beginS,endS,beginV);
      else
         f = new FUNCTION(beginS,endS);
      return f;
   }


   template<class FUNCTION>
   FUNCTION * differenceFunctionConstructor(boost::python::object shape,const typename FUNCTION::ValueType weight){
      FUNCTION * f = NULL;
      stl_input_iterator<int> begin(shape), end;
      const int s1=*begin;
      ++begin;
      const int s2=*begin;
      ++begin;
      f = new FUNCTION(s1,s2,weight);
      return f;
   }

   template<class FUNCTION>
   FUNCTION * truncatedDifferenceFunctionConstructor(boost::python::object shape,const typename FUNCTION::ValueType truncate,const typename FUNCTION::ValueType weight){
      FUNCTION * f = NULL;
      stl_input_iterator<int> begin(shape), end;
      const int s1=*begin;
      ++begin;
      const int s2=*begin;
      ++begin;
      f = new FUNCTION(s1,s2,truncate,weight);
      return f;
   }

   ////////////////////////////////////////
   // EXPLICIT FUNCTION
   ////////////////////////////////////////
   template<class FUNCTION>
   FUNCTION * explicitFunctionConstructorPyAny(boost::python::object shape,const typename FUNCTION::ValueType value){
      stl_input_iterator<int> begin(shape), end;
      return new FUNCTION(begin,end,value);
   }
   
   ////////////////////////////////////////
   // SPARSE FUNCTION
   ////////////////////////////////////////

   template<class FUNCTION>
   FUNCTION * sparseFunctionConstructorPyAny(boost::python::object shape,const typename FUNCTION::ValueType value){
      stl_input_iterator<int> begin(shape), end;
      std::vector<int> vec(begin,end);
      return new FUNCTION(vec.begin(),vec.end(),value);
   }


   
   template<class FUNCTION,class PY_TYPE>
   void sparseFunctionInsertItemList(FUNCTION & f,boost::python::list coordinate,const typename FUNCTION::ValueType value){
      typedef typename FUNCTION::ValueType ValueType;
      typedef typename FUNCTION::LabelType LabelType;
      IteratorHolder< PythonIntListAccessor<PY_TYPE ,true> > holder(coordinate);
      if(std::abs(value-f.defaultValue())>=static_cast<ValueType>(0.0000001)){
         f.insert(holder.begin(),value);
      }
   }

   template<class FUNCTION,class PY_TYPE>
   void sparseFunctionInsertItemTuple(FUNCTION & f,boost::python::tuple coordinate,const typename FUNCTION::ValueType value){
      typedef typename FUNCTION::ValueType ValueType;
      typedef typename FUNCTION::LabelType LabelType;
      IteratorHolder< PythonIntTupleAccessor<PY_TYPE ,true> > holder(coordinate);
      if(std::abs(value-f.defaultValue())>=static_cast<ValueType>(0.0000001)){
         f.insert(holder.begin(),value);
      }
   }

   template<class FUNCTION,class PY_TYPE>
   void sparseFunctionInsertItemNumpy(FUNCTION & f,opengm::python::NumpyView<PY_TYPE,1> coordinate,const typename FUNCTION::ValueType value){
      typedef typename FUNCTION::ValueType ValueType;
      typedef typename FUNCTION::LabelType LabelType;
      if(std::abs(value-f.defaultValue())>=static_cast<ValueType>(0.0000001)){
         f.insert(coordinate.begin(),value);
      }
   }

   template<class FUNCTION>
   inline const typename FUNCTION::ContainerType &
   sparseFunctionConstContainer(const FUNCTION & f){
      return f.container();
   }

   template<class FUNCTION>
   typename FUNCTION::ContainerType &
   sparseFunctionContainer(FUNCTION & f){
      return f.container();
   }

   template<class FUNCTION>
   void keyToCoordinate(const FUNCTION & f ,const typename FUNCTION::KeyType key,opengm::python::NumpyView<typename FUNCTION::LabelType,1> output){
      f.keyToCoordinate(key,output.begin());
   }

   template<class FUNCTION>
   typename FUNCTION::KeyType  coordinateToKey_PyAny ( 
      const FUNCTION & f,
      boost::python::object coordinate
   ){
      stl_input_iterator<int> begin(coordinate), end;
      return f.coordinateToKey(begin);
   }




}

#define FUNCTION_TYPE_EXPORTER_HELPER(CLASS_NAME,CLASS_STRING)\
class_<CLASS_NAME > (CLASS_STRING, init<const CLASS_NAME &> ( (arg("other")),"copy constructor" ))\
.def(init<>("empty constructor"))\
.def("__array__",&pyfunction::copyFunctionValuesToNumpyOrder<CLASS_NAME>,"copy the function into a new allocated numpy ndarray")\
.def("_getitem_tuple", &pyfunction::getValuePyTuple<CLASS_NAME,int>, return_value_policy< return_by_value >())\
.def("_getitem_list", &pyfunction::getValuePyList<CLASS_NAME,int>, return_value_policy< return_by_value >())\
.def("_getitem_numpy", &pyfunction::getValuePyNumpy<CLASS_NAME,typename CLASS_NAME::LabelType>, return_value_policy< return_by_value >())\
.add_property("size", &CLASS_NAME::size,"get the size/number of elements of the function")\
.add_property("dimension", &CLASS_NAME::dimension,"get the number of dimensions")\
.add_property("ndim", &CLASS_NAME::dimension,"get the number of dimensions (same as dimension)")\
.add_property("shape",&pyfunction::getShapeCallByReturnPyTuple< CLASS_NAME,int >,"get the shape of the function")

template<class FUNCTION_TYPE>
void export_function_type_vector(const std::string & className){
    typedef std::vector<FUNCTION_TYPE> PyFunctionTypeVector;
    
    boost::python::class_<PyFunctionTypeVector > (className.c_str())
     .def(boost::python::vector_indexing_suite<PyFunctionTypeVector > ())
   ;

}

template<class VEC>
inline int genericVectorSize(
   const VEC & vec
){
   return vec.size();
}


#define EXPORT_FUNCTION_TYPE_VECTOR(FUNCTION_TYPE,CLASS_NAME) \
    boost::python::class_< std::vector<FUNCTION_TYPE>  > (CLASS_NAME,init<>()) \
     .def(boost::python::vector_indexing_suite< std::vector<FUNCTION_TYPE>  > ()) 

namespace pyfuncvec{
   template<class FUNCTION_TYPE>
   std::vector<FUNCTION_TYPE> * 
   constructPottsFunctionVector(
      opengm::python::NumpyView< typename FUNCTION_TYPE::LabelType,1> numLabels1,
      opengm::python::NumpyView< typename FUNCTION_TYPE::LabelType,1> numLabels2, 
      opengm::python::NumpyView< typename FUNCTION_TYPE::ValueType,1> valuesEqual, 
      opengm::python::NumpyView< typename FUNCTION_TYPE::ValueType,1> valuesNotEqual 
   ){
      typedef typename FUNCTION_TYPE::LabelType LabelType;
      typedef typename FUNCTION_TYPE::ValueType ValueType;
      // number of functions?
      const size_t nL1=numLabels1.shape(0);
      const size_t nL2=numLabels2.shape(0);
      const size_t nVE=valuesEqual.shape(0);
      const size_t nVNE=valuesNotEqual.shape(0);
      const size_t numFunctions=std::max( std::max(nL1,nL2),std::max(nVE,nVNE) );
      // allocate vector
      std::vector<FUNCTION_TYPE> *  vec=new std::vector<FUNCTION_TYPE>(numFunctions);
      // generate functions
      for(size_t f=0;f<numFunctions;++f){
         const LabelType la         =  f < nL1-1   ? numLabels1(f)      : numLabels1(nL1-1);
         const LabelType lb         =  f < nL2-1   ? numLabels2(f)      : numLabels2(nL2-1);
         const ValueType vEqual     =  f < nVE-1   ? valuesEqual(f)     : valuesEqual(nVE-1);
         const ValueType vNotEqual  =  f < nVNE-1  ? valuesNotEqual(f)  : valuesNotEqual(nVNE-1);
         (*vec)[f]=FUNCTION_TYPE(la,lb,vEqual,vNotEqual);
      }
      return vec;
   }
}

template<class V,class I>
void export_functiontypes(){
   import_array();
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   typedef V ValueType;
   typedef I IndexType;
   typedef IndexType LabelType;

   // different function types
   typedef opengm::ExplicitFunction                      <ValueType,IndexType,LabelType> PyExplicitFunction;
   typedef opengm::PottsFunction                         <ValueType,IndexType,LabelType> PyPottsFunction;
   typedef opengm::PottsNFunction                        <ValueType,IndexType,LabelType> PyPottsNFunction;
   typedef opengm::PottsGFunction                        <ValueType,IndexType,LabelType> PyPottsGFunction;
   typedef opengm::AbsoluteDifferenceFunction            <ValueType,IndexType,LabelType> PyAbsoluteDifferenceFunction;
   typedef opengm::TruncatedAbsoluteDifferenceFunction   <ValueType,IndexType,LabelType> PyTruncatedAbsoluteDifferenceFunction;
   typedef opengm::SquaredDifferenceFunction             <ValueType,IndexType,LabelType> PySquaredDifferenceFunction;
   typedef opengm::TruncatedSquaredDifferenceFunction    <ValueType,IndexType,LabelType> PyTruncatedSquaredDifferenceFunction;
   typedef opengm::SparseFunction                        <ValueType,IndexType,LabelType> PySparseFunction; 
   typedef opengm::python::PythonFunction                <ValueType,IndexType,LabelType> PyPythonFunction; 
    
   // vector exporters
   export_function_type_vector<PyExplicitFunction>("ExplicitFunctionVector");
   
   EXPORT_FUNCTION_TYPE_VECTOR(PyPottsFunction,"PottsFunctionVector")
   .def("__init__", make_constructor(&pyfuncvec::constructPottsFunctionVector<PyPottsFunction> ,default_call_policies(),
         (
            boost::python::arg("numberOfLabels1"),
            boost::python::arg("numberOfLabels2"),
            boost::python::arg("valueEqual"),
            boost::python::arg("valueNotEqual")
         )
      ),
      "TODO"
   )
   ;

   export_function_type_vector<PyPottsNFunction>("PottsNFunctionVector");
   export_function_type_vector<PyPottsGFunction>("PottsGFunctionVector");
   //export_function_type_vector<PyAbsoluteDifferenceFunction>("AbsoluteDifferenceFunctionVector");
   export_function_type_vector<PyTruncatedAbsoluteDifferenceFunction>("TruncatedAbsoluteDifferenceFunctionVector");
   //export_function_type_vector<PySquaredDifferenceFunction>("SquaredDifferenceFunctionVector");
   export_function_type_vector<PyTruncatedSquaredDifferenceFunction>("TruncatedSquaredDifferenceFunctionVector");
   export_function_type_vector<PySparseFunction>("SparseFunctionVector");
   export_function_type_vector<PyPythonFunction>("PythonFunctionVector");

   typedef typename PySparseFunction::ContainerType PySparseFunctionMapType;
   //export std::map for sparsefunction
   class_<PySparseFunctionMapType >("SparseFunctionMap")
      .def(map_indexing_suite<PySparseFunctionMapType>() );


   FUNCTION_TYPE_EXPORTER_HELPER(PyExplicitFunction,"ExplicitFunction")
   .def("__init__", make_constructor(&pyfunction::explicitFunctionConstructorPyAny<PyExplicitFunction>,default_call_policies(),
      (
         boost::python::arg("shape"),
         boost::python::arg("value")=0.0
      )
   ),
   "Construct an explicit function from shape and an optional value to fill the function with\n\n"
   "Args : \n\n"
   "   shape : shape of the function \n\n"
   "   value : value to fill the function with (default : 0.0)\n\n"
   "Examples:\n\n"
   "   >>> import opengm\n"
   "   >>> f=opengm.ExplicitFunction(shape=[2,3,4],1.0)\n\n"
   "\n\n"
   "Notes :\n\n"
   "   Instead  of adding an explicit function directly to the graphical model  one can add a numpy ndarray to the gm,\n"
   "   which will be converted to an explicit function. But it might be faster to add the explicit function directly."
   )
   ;   

   FUNCTION_TYPE_EXPORTER_HELPER(PySparseFunction,                      "SparseFunction")
   .def("__init__", make_constructor(&pyfunction::sparseFunctionConstructorPyAny<PySparseFunction>,default_call_policies(),
      (
         boost::python::arg("shape"),
         boost::python::arg("defaultValue")=0.0
      )),
   "Construct a sparse function from shape and an optional value to fill the function with\n\n"
   "Args : \n\n"
   "   shape : shape of the function \n\n"
   "   defaultValue : default value of the sparse function (default : 0.0)\n\n"
   "Examples:\n\n"
   "   >>> import opengm\n"
   "   >>> f=opengm.SparseFunction(shape=[2,3,4],0.0)\n"
   "   >>> len(f.container)\n"
   "   0\n"
   "   >>> f[0,1,0]=1.0\n"
   "   >>> len(f.container)\n"
   "   0\n"
   "\n\n"
   "Notes :\n\n"
   "   Instead  of adding an explicit function directly to the graphical model  one can add a numpy ndarray to the gm,\n"
   "   which will be converted to an explicit function. But it might be faster to add the explicit function directly."
   )
   .def("_setitem",& pyfunction::sparseFunctionInsertItemTuple<PySparseFunction,int>)
   .def("_setitem",& pyfunction::sparseFunctionInsertItemList<PySparseFunction,int>)
   .def("_setitem",& pyfunction::sparseFunctionInsertItemNumpy<PySparseFunction,LabelType>)
   .def("_defaultValue",&PySparseFunction::defaultValue,"get the default value of the sparse function")
   .def("_container",&pyfunction::sparseFunctionContainer<PySparseFunction>,return_internal_reference<>())
   .def("_coordinateToKey",pyfunction::coordinateToKey_PyAny<PySparseFunction>)
   .def("_keyToCoordinateCpp",&pyfunction::keyToCoordinate<PySparseFunction>)
   ;

   FUNCTION_TYPE_EXPORTER_HELPER(PyPottsFunction,                       "PottsFunction")
   .def("__init__", make_constructor(&pyfunction::pottsFunctionConstructor<PyPottsFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("valueEqual"),
            boost::python::arg("valueNotEqual")
         )
      ),
   "Construct a PottsFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  valueEqual: value of the functions where labels are equal (on diagnal of the function as ``f[0,0]``,``f[1,1]``)\n\n"
   "  valueNotEqual: value of the functions where labels differ (off diagnal of the function as ``f[1,0]``,``f[0,1]``)\n\n"
   "Example:\n\n"
   "     Construct a PottsFunction ::\n\n"
   "        >>> f=opengm.PottsFunction([2,2],1.0,0.0)\n"
   "        >>> f[0,0]\n"
   "        0.0\n"
   "        >>> f[0,1]\n"
   "        1.0\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.PottsFunctionVector`"
   )
   ;

   FUNCTION_TYPE_EXPORTER_HELPER(PyPottsNFunction,                      "PottsNFunction")
   .def("__init__", make_constructor(&pyfunction::pottsNFunctionConstructor<PyPottsNFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("valueEqual"),
            boost::python::arg("valueNotEqual")
         )
      ),
   "Construct a PottsNFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  valueEqual: value of the functions where labels are equal (on diagnal of the function as ``f[0,0]``,``f[1,1]``)\n\n"
   "  valueNotEqual: value of the functions where labels differ (off diagnal of the function as ``f[1,0]``,``f[0,1]``)\n\n"
   "Example:\n\n"
   "     Construct a PottsFunction ::\n\n"
   "        >>> f=opengm.PottsNFunction([4,4,4],1.0,0.0)\n"
   "        >>> f[3,3,3]\n"
   "        0.0\n"
   "        >>> f[0,1,1]\n"
   "        1.0\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.PottsFunctionVector`"
   )
   ;

   FUNCTION_TYPE_EXPORTER_HELPER(PyPottsGFunction,                      "PottsGFunction")
   .def("__init__", make_constructor(&pyfunction::pottsGFunctionConstructor<PyPottsGFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("values")=boost::python::make_tuple()
         )
      ),
   "Construct a PottsGFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  values:  TODO!!!!! \n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.PottsGFunctionVector`"
   )
   ;

   /*
   FUNCTION_TYPE_EXPORTER_HELPER(PyAbsoluteDifferenceFunction,          "AbsoluteDifferenceFunction")
   .def("__init__", make_constructor(&pyfunction::differenceFunctionConstructor<PyAbsoluteDifferenceFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("weight")=1.0
         )
      ),
   "Construct a AbsoluteDifferenceFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  weight: weight of the function (default : 1.0) \n\n"
   "Example:\n\n"
   "   >>> f=opengm.AbsoluteDifferenceFunction([255,255],2.0)\n"
   "   >>> f[200,100]== abs(200.0 -100.0 )*2.0\n"
   "   0.0\n"
   "   >>> f[0,1,1]\n"
   "   1.0\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.AbsoluteDifferenceFunctionVector`"
   )
   ;
   */

   FUNCTION_TYPE_EXPORTER_HELPER(PyTruncatedAbsoluteDifferenceFunction, "TruncatedAbsoluteDifferenceFunction")
   .def("__init__", make_constructor(&pyfunction::truncatedDifferenceFunctionConstructor<PyTruncatedAbsoluteDifferenceFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("truncate"),
            boost::python::arg("weight")=1.0
         )
      ),
   "Construct a TruncatedAbsoluteDifferenceFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  truncate : truncate the function at a given value \n\n"
   "  weight: weight of the function (default : 1.0) \n\n"
   "Example: ::\n\n"
   "   >>> f=opengm.TruncatedAbsoluteDifferenceFunction(shape=[255,255],truncate=20.0,weight=2.0)\n\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.TruncatedAbsoluteDifferenceFunctionVector`"
   )
   ;
   /*
   FUNCTION_TYPE_EXPORTER_HELPER(PySquaredDifferenceFunction,           "SquaredDifferenceFunction")
   .def("__init__", make_constructor(&pyfunction::differenceFunctionConstructor<PySquaredDifferenceFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("weight")=1.0
         )
      ),
   "Construct a SquaredDifferenceFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  weight: weight of the function (default : 1.0) \n\n"
   "Example: ::\n\n"
   "   >>> f=opengm.TruncatedAbsoluteDifferenceFunction(shape=[255,255],truncate=20.0,weight=2.0)\n\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.AbsoluteDifferenceFunctionVector`"
   )
   ;
   */
   
   FUNCTION_TYPE_EXPORTER_HELPER(PyTruncatedSquaredDifferenceFunction,  "TruncatedSquaredDifferenceFunction")
   .def("__init__", make_constructor(&pyfunction::truncatedDifferenceFunctionConstructor<PyTruncatedSquaredDifferenceFunction> ,default_call_policies(),
         (
            boost::python::arg("shape"),
            boost::python::arg("truncate"),
            boost::python::arg("weight")=1.0
         )
      ),
   "Construct a TruncatedSquaredDifferenceFunction .\n\n"
   "Args:\n\n"
   "  shape: shape of the function (len(shape) must be 2 !)\n\n"
   "  truncate : truncate the function at a given value \n\n"
   "  weight: weight of the function (default : 1.0) \n\n"
   "Example: ::\n\n"
   "   >>> f=opengm.TruncatedSquaredDifferenceFunction(shape=[255,255],truncate=20.0,weight=2.0)\n\n"
   "\n\n"
   "Note:\n\n"
   ".. seealso::"
   "     :func:`opengm.TruncatedSquaredDifferenceFunctionVector`"
   )
   ;
   
   FUNCTION_TYPE_EXPORTER_HELPER(PyPythonFunction,                       "PythonFunction")
   .def(init<boost::python::object,boost::python::object,const bool>(
         (arg("function"),arg("shape"),arg("ensureGilState")=true),
         "Examples: ::\n\n"
         "   >>> import opengm\n"
         "   >>> import numpy\n" 
         "   >>> def labelSumFunction(labels):\n"
         "   ...    s=0\n"
         "   ...    for l in labels:\n"
         "   ...       s+=l\n"
         "   ...    return s\n"
         "   >>> f=opengm.PythonFunction(function=labelSumFunction,shape=[2,2])\n"
         "\n\n"
      )
   )
   ;

}

template void export_functiontypes<opengm::python::GmValueType,opengm::python::GmIndexType>();


