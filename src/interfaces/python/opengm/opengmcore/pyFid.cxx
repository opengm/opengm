#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <stdexcept>
#include <stddef.h>

#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

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
           .add_property("functionType", &PyFid::getFunctionType)
           .add_property("functionIndex", &PyFid::getFunctionIndex)
           
   ;

}

template void export_fid<opengm::python::GmIndexType>();