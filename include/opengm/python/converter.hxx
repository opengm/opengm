#ifndef CONVERTER_IMPL_HXX
#define CONVERTER_IMPL_HXX

#include <boost/python/detail/wrap_python.hpp>
#include <boost/python.hpp>

#include <sstream>
//#include <Python.h>
#include <numpy/arrayobject.h>

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>



#include <stdexcept>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/python/numpyview.hxx>

namespace opengm{
namespace python{

using namespace boost::python;



   template <typename T>
   inline NPY_TYPES typeEnumFromType(void) {
   PyErr_SetString(PyExc_ValueError, "no mapping available for this type");
   boost::python::throw_error_already_set();
   return NPY_VOID;
   }

   //NPY_BOOL
   template <> inline NPY_TYPES typeEnumFromType<bool>(void) {
   return NPY_BOOL;
   }


   template <> inline NPY_TYPES typeEnumFromType<opengm::UInt8Type>(void) {
   return NPY_UINT8;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::UInt16Type>(void) {
   return NPY_UINT16;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::UInt32Type>(void) {
   return NPY_UINT32;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::UInt64Type>(void) {
   return NPY_UINT64;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::Int8Type>(void) {
   return NPY_INT8;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::Int16Type>(void) {
   return NPY_INT16;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::Int32Type>(void) {
   return NPY_INT32;
   }

   template <> inline NPY_TYPES typeEnumFromType<opengm::Int64Type>(void) {
   return NPY_INT64;
   }

   template <> inline NPY_TYPES typeEnumFromType<float>(void) {
   return NPY_FLOAT32;
   }

   template <> inline NPY_TYPES typeEnumFromType<double>(void) {
   return NPY_FLOAT64;
   }




template<class VALUE_TYPE>
inline boost::python::object get1dArray(const size_t size){
   npy_intp dims[1]={static_cast<int>(size)};
   boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(int(1),  dims, typeEnumFromType<VALUE_TYPE>() )));
   return obj;
}

template<class VALUE_TYPE>
inline boost::python::object get2dArray(const size_t size1,const size_t size2){
   npy_intp dims[2]={static_cast<int>(size1),static_cast<int>(size2)};
   boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(int(2),  dims, typeEnumFromType<VALUE_TYPE>() )));
   return obj;
}


template<class VALUE_TYPE,class FORWARD_SHAPE_ITERATOR>
inline boost::python::object getArray(FORWARD_SHAPE_ITERATOR begin,FORWARD_SHAPE_ITERATOR end){
   const int nDim=std::distance(begin,end);
   npy_intp * dims = new npy_intp[nDim];
   std::copy(begin,end,dims);
   boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(nDim,  dims, typeEnumFromType<VALUE_TYPE>() )));
   delete[] dims;
   return obj;
}

template<class VALUE_TYPE>
inline VALUE_TYPE * getCastedPtr(boost::python::object obj){
   void *array_data = PyArray_DATA((PyArrayObject*) obj.ptr());
   return  static_cast< VALUE_TYPE *>(array_data);
}

inline boost::python::numeric::array objToArray(boost::python::object obj){
   return boost::python::extract<boost::python::numeric::array > (obj);
}




inline std::string printEnum(PyArray_TYPES value) {
   if (value == PyArray_UBYTE) return std::string("PyArray_UBYTE");
   else if (value == PyArray_BOOL) return std::string("PyArray_BOOL");
   else if (value == PyArray_UINT8) return std::string("PyArray_UINT8");
   else if (value == PyArray_UINT16) return std::string("PyArray_UINT16");
   else if (value == PyArray_UINT32) return std::string("PyArray_UINT32");
   else if (value == PyArray_UINT64) return std::string("PyArray_UINT64");
   else if (value == PyArray_INT8) return std::string("PyArray_INT8");
   else if (value == PyArray_INT16) return std::string("PyArray_INT16");
   else if (value == PyArray_INT32) return std::string("PyArray_INT32");
   else if (value == PyArray_INT64) return std::string("PyArray_INT64");
   else if (value == PyArray_FLOAT32) return std::string("PyArray_FLOAT32");
   else if (value == PyArray_FLOAT64) return std::string("PyArray_FLOAT64");
   else if (value == PyArray_BYTE) return std::string("PyArray_BYTE");
   else if (value == PyArray_UBYTE) return std::string("PyArray_UBYTE");
   else if (value == PyArray_USHORT) return std::string("PyArray_USHORT");
   else if (value == PyArray_INT) return std::string("PyArray_INT");
   else if (value == PyArray_UINT) return std::string("PyArray_UINT");
   else if (value == PyArray_LONG) return std::string("PyArray_LONG");
   else if (value == PyArray_ULONG) return std::string("PyArray_ULONG");
   else if (value == PyArray_LONGLONG) return std::string("PyArray_LONGLONG");
   else if (value == PyArray_DOUBLE) return std::string("PyArray_DOUBLE");
   else if (value == PyArray_LONGDOUBLE) return std::string("PyArray_LONGDOUBLE");
   else if (value == PyArray_FLOAT) return std::string("PyArray_FLOAT");
   else if (value == PyArray_CFLOAT) return std::string("PyArray_CFLOAT");
   else if (value == PyArray_CDOUBLE) return std::string("PyArray_CDOUBLE");

   else return " unkown type";
}

template<class ITERATOR>
inline boost::python::tuple iteratorToTuple(ITERATOR iter, size_t size) {
   typedef typename std::iterator_traits<ITERATOR>::value_type ValueType;
   PyObject* tuple = PyTuple_New(size);
   if (opengm::meta::Compare<ValueType, opengm::UInt8Type>::value ||
      opengm::meta::Compare<ValueType, opengm::UInt16Type>::value ||
      opengm::meta::Compare<ValueType, opengm::UInt32Type>::value ||
      opengm::meta::Compare<ValueType, opengm::Int8Type>::value ||
      opengm::meta::Compare<ValueType, opengm::Int16Type>::value ||
      opengm::meta::Compare<ValueType, opengm::Int32Type>::value
   ) {
      for (size_t i = 0; i<size; ++i) {
         PyTuple_SetItem(tuple, i, PyInt_FromLong(long(iter[i])));
      }
   }
   else if (opengm::meta::Compare<ValueType, opengm::UInt64Type>::value ||
      opengm::meta::Compare<ValueType, opengm::Int64Type>::value
   ) {
      for (size_t i = 0; i<size; ++i) {
         PyTuple_SetItem(tuple, i, PyLong_FromLong(long(iter[i])));
      }
   }
   else if (opengm::meta::Compare<ValueType, opengm::Float32Type>::value ||
      opengm::meta::Compare<ValueType, opengm::Float64Type>::value) {
      for (size_t i = 0; i<size; ++i) {
         PyTuple_SetItem(tuple, i, PyFloat_FromDouble(double(iter[i])));
      }
   }
   else {
      opengm::RuntimeError("selected type is not supported in this to-tuple converter ");
   }
   boost::python::tuple t = boost::python::extract<boost::python::tuple > (tuple);
   return t;
}

template<class ITERATOR>
inline boost::python::list iteratorToList(ITERATOR iter, size_t size) {
   typedef typename std::iterator_traits<ITERATOR>::value_type ValueType;
   boost::python::list l;
   for(size_t i=0;i<size;++i)
      l.append(iter[i]);
   return l;
}

template<class ITERATOR>
inline boost::python::numeric::array iteratorToNumpy(ITERATOR iter, size_t size) {
   typedef typename std::iterator_traits<ITERATOR>::value_type ValueType;
   boost::python::object obj = get1dArray<ValueType>(size);
   ValueType * castedPtr = getCastedPtr<ValueType>(obj);
   for(size_t i=0;i<size;++i)
      castedPtr[i]=iter[i];
   return objToArray(obj);
}

template<class NUMERIC_ARRAY>
inline PyArray_TYPES getArrayType(NUMERIC_ARRAY arr) {
   return PyArray_TYPES(PyArray_TYPE(arr.ptr()));
}

inline boost::python::numeric::array extractConstNumericArray
(
   PyObject * obj
) {
   return boost::python::extract<boost::python::numeric::array > (obj);
}

inline int numpyScalarTypeNumber(PyObject* obj) {
   PyArray_Descr* dtype;
   if (!PyArray_DescrConverter(obj, &dtype)) return NPY_NOTYPE;
   int typeNum = dtype->type_num;
   Py_DECREF(dtype);
   return typeNum;
}

template<class VALUE_TYPE,size_t DIM>
struct NumpyViewType_from_python_numpyarray {
   typedef VALUE_TYPE ValueType;
   typedef NumpyView <ValueType,DIM> NumpyViewType;

   NumpyViewType_from_python_numpyarray() {
      boost::python::converter::registry::push_back(&convertible, &construct, boost::python::type_id<NumpyViewType > ());
   }

   // Determine if obj_ptr can be converted in a NumpyViewType

   static void* convertible(PyObject * obj_ptr) {
      if (!PyArray_Check(obj_ptr)) {
         //PyErr_SetString(PyExc_ValueError, "expected a PyArrayObject");
         return 0;
      } else {
         numeric::array numpyArray = extractConstNumericArray(obj_ptr);
         PyArray_TYPES pyArrayType = getArrayType(numpyArray);
         // check if the type of the numpy array matches the c++ type  
         PyArray_TYPES myEnum = typeEnumFromType<ValueType > ();
         if (myEnum != pyArrayType) {
            std::stringstream ss;
            ss << "type mismatch:\n";
            ss << "python type: " << printEnum(pyArrayType) << "\n";
            ss << "c++ expected type : " << printEnum(myEnum);
            PyErr_SetString(PyExc_ValueError, ss.str().c_str());
            return 0;
         }
         if(DIM!=0){
            const boost::python::tuple &shape = boost::python::extract<boost::python::tuple > (numpyArray.attr("shape"));
            if(boost::python::len(shape)!=DIM){
               std::stringstream ss;
               ss << "dimension mismatch:\n";
               ss << "python numpy dimension         : " << boost::python::len(shape) << "\n";
               ss << "c++  expected  dimension : " << DIM;
               PyErr_SetString(PyExc_ValueError, ss.str().c_str());   
            }
         }
      }
      return obj_ptr;
   }

   // Convert obj_ptr into a NumpyViewType

   static void construct(
      PyObject* obj_ptr,
      boost::python::converter::rvalue_from_python_stage1_data * data) {
      // Extract the character data from the python string

      // Grab pointer to memory into which to construct the new NumpyViewType
      void* storage = (
         (boost::python::converter::rvalue_from_python_storage<NumpyViewType>*)
         data)->storage.bytes;

      // in-place construct the new NumpyViewType using the character data
      // extraced from the python object
      //const numeric::array & numpyArray = extractConstNumericArray(obj_ptr);
      new (storage) NumpyViewType(boost::python::object(boost::python::borrowed(obj_ptr)));
      // Stash the memory chunk pointer for later use by boost.python
      data->convertible = storage;
   }
};


template<class VALUE_TYPE,size_t DIM>
struct NumpyViewType_to_python_numpyarray{
   
   typedef NumpyView<VALUE_TYPE,DIM> NumpyViewType;
   
   static PyObject * convert(NumpyViewType  numpyView ){
      return numpyView.object().ptr();
   }
};


template<class T,size_t DIM>
void initializeNumpyViewConverters() {
   using namespace boost::python;
   NumpyViewType_to_python_numpyarray<T ,DIM> ();
   NumpyViewType_from_python_numpyarray<T ,DIM> ();
}





}
}


#endif   

