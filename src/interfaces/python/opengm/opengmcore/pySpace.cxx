#include <boost/python.hpp>
#include <stdexcept>
#include <string>
#include <sstream>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>
#include <opengm/python/pythonfunction.hxx>



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
   
   class_<PySpace > ("Space", 
   "The variable space of a graphical model.\n\n"
   "Stores the number of variables and the number of labels for each variable",
   init< >()
   )
   .def("__str__",&pyspace::asString<PySpace>)
   .add_property("size", &PySpace::numberOfVariables)
   .add_property("numberOfVariables", &PySpace::numberOfVariables,
   "Get the number of variables in the variable spaec.\n\n"
   "Returns:\n"
   "  Number of variables\n\n"
   )
   .def("__len__",&PySpace::numberOfVariables,
   "Get the number of variables in the variable spaec.\n\n"
   "Returns:\n"
   "  Number of variables\n\n"
   )
   .def("__getitem__", &PySpace::numberOfLabels, return_value_policy<return_by_value>(),(arg("variableIndexs")),
   "Get the number of variables in the variable space.\n\n"
   "Args:\n\n"
   "  variableIndex: maximum subgraph size which is optimized\n\n"
   "Returns:\n"
   "  number of labels for the variable at ``variableIndex``"
   )
   ;
}


template void export_space<opengm::python::GmIndexType>();

