#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include "export_typedes.hxx"


using namespace boost::python;

namespace pyspace {
   template<class SPACE>
   std::string asString(const SPACE & space){
      std::stringstream ss;
      for(size_t i=0;i<space.numberOfVariables();++i){
         if(i==space.numberOfVariables()-1)
            ss<<"vi="<<i<<", number of labels="<<space.numberOfLabels(i);
         else
            ss<<"vi="<<i<<", number of labels="<<space.numberOfLabels(i)<<"\n";
      }
      return ss.str();
   }
}



template<class INDEX_TYPE>
void export_space() {

   typedef INDEX_TYPE IndexType;
   typedef opengm::DiscreteSpace<IndexType,IndexType> PySpace;
   
   class_<PySpace > ("Space", init< >())
   .def("__str__",&pyspace::asString<PySpace>)
   .add_property("size", &PySpace::numberOfVariables)
   .add_property("numberOfVariables", &PySpace::numberOfVariables)
   .def("__len__",&PySpace::numberOfVariables)
   .def("__getitem__", &PySpace::numberOfLabels, return_value_policy<return_by_value>())
   ;
}


template void export_space<GmIndexType>();

