#ifndef PY_OPENGM_HDF5
#define PY_OPENGM_HDF5

// #define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleHdf5




#ifdef WITH_HDF5
#include <boost/python.hpp>
#include <stdexcept>
#include <stddef.h>
#include <opengm/graphicalmodel/graphicalmodel.hxx>
#include <opengm/graphicalmodel/graphicalmodel_hdf5.hxx>
#include <opengm/python/opengmpython.hxx>
#include <opengm/python/converter.hxx>
#include <opengm/python/numpyview.hxx>

//using namespace boost::python;

template<class GM>
void export_hdf5() {
   
   

   typedef GM PyGm;
    // export stuff in the hdf5 namespace
    //bp::class_<Util::String>("String");
    // etc.
   
   //------------------------------------------------------------------------------------
   // function to do hdf5 io
   //------------------------------------------------------------------------------------
   boost::python::def("saveGraphicalModel",
      opengm::hdf5::save<PyGm>,
      (
      boost::python::arg("gm"),
      boost::python::arg("file"),
      boost::python::arg("dataset")
      ),
      "saveGraphicalModel"
      );
   boost::python::def("loadGraphicalModel",
      opengm::hdf5::load<PyGm>,
      (
      boost::python::arg("gm"),
      boost::python::arg("file"),
      boost::python::arg("dataset")
      ),
      "loadGraphicalModel"
      );
}

template void export_hdf5<opengm::python::GmAdder>();


#endif //WITH_HDF5
#endif