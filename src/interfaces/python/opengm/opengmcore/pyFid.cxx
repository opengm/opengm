#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCore

#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "export_typedes.hxx"

using namespace boost::python;

template<class INDEX_TYPE>
void export_fid(){
   typedef INDEX_TYPE FunctionIndexType;
   typedef opengm::UInt8Type FunctionTypeIndexType;
   typedef opengm::FunctionIdentification<FunctionIndexType,FunctionTypeIndexType> PyFid;
   //------------------------------------------------------------------------------------
   // function identifier
   //------------------------------------------------------------------------------------
   class_<PyFid > ("FunctionIdentifier", init<const FunctionIndexType, const FunctionTypeIndexType > ())
           .def("getFunctionType", &PyFid::getFunctionType)
           .def("getFunctionIndex", &PyFid::getFunctionIndex)
   ;

}

template void export_fid<GmIndexType>();