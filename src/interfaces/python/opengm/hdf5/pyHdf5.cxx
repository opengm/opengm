#ifndef PY_OPENGM_HDF5
#define PY_OPENGM_HDF5

#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleHdf5


#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#ifdef WITH_HDF5
#include <stdexcept>
#include <stddef.h>
#include <boost/python.hpp>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include "../export_typedes.hxx"

using namespace boost::python;

template<class GM>
void export_hdf5() {
   
   

   typedef GM PyGm;
    // export stuff in the hdf5 namespace
    //bp::class_<Util::String>("String");
    // etc.
   
   //------------------------------------------------------------------------------------
   // function to do hdf5 io
   //------------------------------------------------------------------------------------
   def("saveGraphicalModel",
      opengm::hdf5::save<PyGm>,
      (
      arg("gm"),
      arg("file"),
      arg("dataset")
      ),
      "saveGraphicalModel"
      );
   def("loadGraphicalModel",
      opengm::hdf5::load<PyGm>,
      (
      arg("gm"),
      arg("file"),
      arg("dataset")
      ),
      "loadGraphicalModel"
      );
}

template void export_hdf5<GmAdder>();


#endif //WITH_HDF5

#endif