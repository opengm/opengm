#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleHdf5

#ifndef OPENGM_PYTHON_INTERFACE
#define OPENGM_PYTHON_INTERFACE 1
#endif

#include <stddef.h>
#include <boost/python.hpp>

#ifdef WITH_HDF5
#include "pyHdf5.hxx"
#endif
#include "export_typedes.hxx"

using namespace boost::python;

BOOST_PYTHON_MODULE_INIT(_hdf5) {
   #ifdef WITH_HDF5
   export_hdf5<GmAdder>();
   #endif
}
