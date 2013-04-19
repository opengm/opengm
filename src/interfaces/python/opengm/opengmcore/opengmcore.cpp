#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleCoreOPENGM



#include <boost/python.hpp>
#include <boost/python/suite/indexing/vector_indexing_suite.hpp>
#include <boost/python/module.hpp>
#include <boost/python/def.hpp>
#include <boost/python/exception_translator.hpp>
#include <stddef.h>
#include <exception>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/utilities/tribool.hxx>
#include <opengm/inference/inference.hxx>
#include "copyhelper.hxx"

#include "pyConfig.hxx"
#include "pyFactor.hxx"
#include "pyMovemaker.hxx"
#include "pyIfactor.hxx" 
#include "pyGm.hxx"     
#include "pyFid.hxx"
#include "pyEnum.hxx"   
#include "pyFunctionTypes.hxx"
#include "pyFunctionGen.hxx"
#include "pySpace.hxx"
#include "pyVector.hxx"
#include "export_typedes.hxx"
#include "converter.hxx"



void translateOpenGmRuntimeError(opengm::RuntimeError const& e)
{
    PyErr_SetString(PyExc_RuntimeError, e.what());
}

using namespace boost::python;



IndexVectorVectorType *
secondOrderGridVis(
   const size_t dx,
   const size_t dy,
   bool order
){
   // calculate the number of factors...
   const size_t hFactors=(dx-1)*dy;
   const size_t vFactors=(dy-1)*dx;
   const size_t numFac=hFactors+vFactors;
   //
   IndexVectorVectorType * vecVec=new IndexVectorVectorType( numFac,IndexVectorType(2));
   size_t fi=0;
   if(order){
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(x+1<dx){
            (*vecVec)[fi][0]=(y+x*dy);
            (*vecVec)[fi][1]=(y+(x+1)*dy);
            ++fi;
         }
         if(y+1<dy){
            boost::python::list vis;
            (*vecVec)[fi][0]=(y+x*dy);
            (*vecVec)[fi][1]=((y+1)+x*dy);
            ++fi;
         }
      }
   }
   else{
      for(size_t x=0;x<dx;++x)
      for(size_t y=0;y<dy;++y){
         if(y+1<dy){
            (*vecVec)[fi][0]=(x+y*dx);
            (*vecVec)[fi][1]=(x+(y+1)*dx);
            ++fi;
         }
         if(x+1<dx){
            boost::python::list vis;
            (*vecVec)[fi][0]=(x+y*dx);
            (*vecVec)[fi][1]=((x+1)+y*dx);
            ++fi;
         }
      }
   }
   return vecVec;
}


BOOST_PYTHON_MODULE_INIT(_opengmcore) {
   Py_Initialize();
   PyEval_InitThreads();
   boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
   boost::python::docstring_options docstringOptions(true,true,false);

   // specify that this module is actually a package
   object package = scope();
   package.attr("__path__") = "opengm";
   
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
   
   currentScopeName="opengm";
   
   class_< opengm::meta::EmptyType > ("_EmptyType",init<>())
   ;

   /*
   typedef ShapeWalkerPython<GmIndexType> PyShapeWalker;

   class_< PyShapeWalker > ("ShapeWalker", "doc",init< NumpyView<GmIndexType,1> >() )
   .def("coordinate", &PyShapeWalker::coordinateTuple, with_custodian_and_ward_postcall<0, 1>(),"get dnarray view to coordinate")
   .def("next",&PyShapeWalker::next,"next coordinate")
   ;
   */











   
   def("secondOrderGridVis", &secondOrderGridVis,return_value_policy<manage_new_object>(),(arg("dimX"),arg("dimY"),arg("numpyOrder")=true),
	"Todo.."
	);
   
   //export_rag();
   export_config();
   export_vectors<GmIndexType>();
   export_space<GmIndexType>();
   export_functiontypes<GmValueType,GmIndexType>();
   export_fid<GmIndexType>();
   export_ifactor<GmValueType,GmIndexType>();
   export_enum();
   export_function_generator<GmAdder,GmMultiplier>();
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
      export_movemaker<GmAdder>();
      
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
      export_movemaker<GmMultiplier>();
   }
   
}
