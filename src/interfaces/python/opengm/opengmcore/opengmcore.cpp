#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

//#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCore


#include <stddef.h>
#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <exception>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/inference.hxx>
#include "opengm_helpers.hxx"
#include "copyhelper.hxx"

#include "pyConfig.hxx"
#include "pyFactor.hxx"
#include "pyIfactor.hxx" 
#include "pyGm.hxx"     
#include "pyFid.hxx"
#include "pyEnum.hxx"   
#include "pyFunctionTypes.hxx"
#include "pySpace.hxx"
#include "pyVector.hxx"   
#include "export_typedes.hxx"
#include "converter.hxx"



void translateOpenGmRuntimeError(opengm::RuntimeError const& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

using namespace boost::python;

void something_which_throws(){
   throw opengm::RuntimeError("damm shot");
}

BOOST_PYTHON_MODULE_INIT(_opengmcore) {
   
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   
   import_array();
   // converters 1d
   initializeNumpyViewConverters<float,1>(); 
   initializeNumpyViewConverters<double,1>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,1>();
   initializeNumpyViewConverters<opengm::UInt64Type,1>();
   initializeNumpyViewConverters<opengm::Int32Type,1>();
   initializeNumpyViewConverters<opengm::Int64Type,1>();
   // converters nd
   initializeNumpyViewConverters<float,0>(); 
   initializeNumpyViewConverters<double,0>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,0>();
   initializeNumpyViewConverters<opengm::UInt64Type,0>();
   initializeNumpyViewConverters<opengm::Int32Type,0>();
   initializeNumpyViewConverters<opengm::Int64Type,0>();
   
   // runtimerror 
  register_exception_translator<opengm::RuntimeError>(&translateOpenGmRuntimeError);
  def("something_which_throws", something_which_throws);
   
   std::string adderString="adder";
   std::string multiplierString="multiplier";
   
   scope current;
   std::string currentScopeName(extract<const char*>(current.attr("__name__")));
   
   export_config();
   export_vectors<GmIndexType>();
   export_space<GmIndexType>();
   export_functiontypes<GmValueType,GmIndexType>();
   export_fid<GmIndexType>();
   export_ifactor<GmValueType,GmIndexType>();
   
   export_enum();
   //adder
   {
      std::string substring=adderString;
      std::string submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      export_gm<GmAdder>();
      export_factor<GmAdder>();
   }
   //multiplier
   {
      std::string substring=multiplierString;
      std::string submoduleName = currentScopeName + std::string(".") + substring;
      // Create the submodule, and attach it to the current scope.
      object submodule(borrowed(PyImport_AddModule(submoduleName.c_str())));
      current.attr(substring.c_str()) = submodule;
      // Switch the scope to the submodule, add methods and classes.
      scope submoduleScope = submodule;
      export_gm<GmMultiplier>();
      export_factor<GmMultiplier>();
   }
   
}
