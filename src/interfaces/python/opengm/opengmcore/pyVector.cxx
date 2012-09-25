#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif
#include <stdexcept>
#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include "../export_typedes.hxx"
#include "../converter.hxx"
using namespace boost::python;


namespace pyvector{



   template<class VECTOR>
           boost::python::list asList(const VECTOR & vector) {
      return iteratorToList(vector.begin(), vector.size());
   }

   template<class VECTOR>
           boost::python::tuple asTuple(const VECTOR & vector) {
      return iteratorToTuple(vector.begin(), vector.size());
   }

   template<class VECTOR>
           boost::python::numeric::array asNumpy(const VECTOR & vector) {
      return iteratorToNumpy(vector.begin(), vector.size());
   }

   template<class VECTOR>
           std::string asString(const VECTOR & vector) {
      std::stringstream ss;
      ss << "(";
      for (size_t i = 0; i < vector.size(); ++i) {
         //if(i!=vector.size()-1)
         ss << vector[i] << ", ";
         //else
         // ss << vector[i];
      }
      ss << ")";
      return ss.str();
   }

}

template<class INDEX>
void export_vectors() {
   import_array();
   typedef std::vector<INDEX> IndexTypeStdVector;
   //------------------------------------------------------------------------------------
   // std vectors 
   //------------------------------------------------------------------------------------
   boost::python::class_<IndexTypeStdVector > ("IndexVector")
           .add_property("size", &IndexTypeStdVector::size)
           .def(boost::python::vector_indexing_suite<IndexTypeStdVector > ())
           .def("__str__", &pyvector::asString<IndexTypeStdVector>)
           .def("asNumpy", &pyvector::asNumpy<IndexTypeStdVector>)
           .def("asTuple", &pyvector::asTuple<IndexTypeStdVector>)
           .def("asList", &pyvector::asList<IndexTypeStdVector>)
           ;

}

template void export_vectors<GmIndexType>();