#define PY_ARRAY_UNIQUE_SYMBOL PyArrayHandleHdf5


#include <boost/python.hpp>
#include <stddef.h>


#ifdef WITH_HDF5
#include "pyHdf5.hxx"
#endif
#include "export_typedes.hxx"

using namespace boost::python;

BOOST_PYTHON_MODULE_INIT(_hdf5) {
    Py_Initialize();
    PyEval_InitThreads();
    #ifdef WITH_HDF5
    export_hdf5<GmAdder>();
    #endif
}
