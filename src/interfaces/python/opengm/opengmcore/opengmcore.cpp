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
#include "pyRegionGraph.hxx"
#ifdef WITH_LIBDAI
#include "inference/external/pyLibdaiEnum.hxx"
#endif   
#include "export_typedes.hxx"
#include "converter.hxx"



void translateOpenGmRuntimeError(opengm::RuntimeError const& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

using namespace boost::python;




boost::python::list secondOrderGridVis(
   const size_t dx,
   const size_t dy,
   bool order
){
   if(order){
      boost::python::list vislist;
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(x+1<dx){
            boost::python::list vis;
            vis.append(y+x*dy);
            vis.append(y+(x+1)*dy);
            vislist.append(vis);
         }
         if(y+1<dy){
            boost::python::list vis;
            vis.append(y+x*dy);
            vis.append((y+1)+x*dy);
            vislist.append(vis);
         }
      }
      return vislist;
   }
   else{
      boost::python::list vislist;
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(y+1<dy){
            boost::python::list vis;
            vis.append(x+y*dx);
            vis.append(x+(y+1)*dx);
            vislist.append(vis);
         }
         if(x+1<dx){
            boost::python::list vis;
            vis.append(x+y*dx);
            vis.append((x+1)+y*dx);
            vislist.append(vis);
         }
      }
      return vislist;
   }
   
}








BOOST_PYTHON_MODULE_INIT(_opengmcore) {
   
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   
   // specify that this module is actually a package
    //object package = scope();
    //package.attr("__path__") = "opengm.opengmcore._opengmcore";
   
   import_array();
   // converters 1d
   initializeNumpyViewConverters<float,1>(); 
   initializeNumpyViewConverters<double,1>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,1>();
   initializeNumpyViewConverters<opengm::UInt64Type,1>();
   initializeNumpyViewConverters<opengm::Int32Type,1>();
   initializeNumpyViewConverters<opengm::Int64Type,1>();
   // converters 2d
   initializeNumpyViewConverters<float,2>(); 
   initializeNumpyViewConverters<double,2>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,2>();
   initializeNumpyViewConverters<opengm::UInt64Type,2>();
   initializeNumpyViewConverters<opengm::Int32Type,2>();
   initializeNumpyViewConverters<opengm::Int64Type,2>();
   // converters 3d
   initializeNumpyViewConverters<float,3>(); 
   initializeNumpyViewConverters<double,3>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,3>();
   initializeNumpyViewConverters<opengm::UInt64Type,3>();
   initializeNumpyViewConverters<opengm::Int32Type,3>();
   initializeNumpyViewConverters<opengm::Int64Type,3>();
   // converters nd
   initializeNumpyViewConverters<float,0>(); 
   initializeNumpyViewConverters<double,0>(); 
   initializeNumpyViewConverters<opengm::UInt32Type,0>();
   initializeNumpyViewConverters<opengm::UInt64Type,0>();
   initializeNumpyViewConverters<opengm::Int32Type,0>();
   initializeNumpyViewConverters<opengm::Int64Type,0>();
   

   
   std::string adderString="adder";
   std::string multiplierString="multiplier";
   
   scope current;
   std::string currentScopeName(extract<const char*>(current.attr("__name__")));
   
   
   
   
   def("secondOrderGridVis", &secondOrderGridVis,(arg("dimX"),arg("dimY"),arg("numpyOrder")=true),
	"Todo.."
	);
   
   export_rag();
   export_config();
   export_vectors<GmIndexType>();
   export_space<GmIndexType>();
   export_functiontypes<GmValueType,GmIndexType>();
   export_fid<GmIndexType>();
   export_ifactor<GmValueType,GmIndexType>();
   #ifdef WITH_LIBDAI
   export_libdai_enums();
   #endif
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
